#!/usr/bin/env python
# coding: utf-8

#######################################################
# IMPORTS
#######################################################
import os,sys
import torch
from torch import nn
from torchvision import transforms
import numpy as np
from torchvision import models as model
from advertorch.utils import NormalizeByChannelMeanStd
from tqdm.notebook import tqdm
from PIL import Image
from plot_image_grid import image_grid
from robustness import model_utils
from robustness.datasets import ImageNet
import matplotlib.pyplot as plt
import random
torch.manual_seed(42)

import pathlib
current_path = pathlib.Path().absolute()

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)

#######################################################
# FUNCTIONS & HELPERS
#######################################################
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def create_folder(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        # directory already exists
        pass

#######################################################
# Initialize Adversarial-Based Optimization
#######################################################
class GradientTexturization():
    def __init__(self,  model_names, n_views, inter_cam, lights, image_size=224):
        super().__init__()

        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_views    = n_views
        self.image_size = image_size
        self.model_names= model_names
        self.inter_cam  = inter_cam
        self.lights     = lights.to(self.device)
        #########################################################
        #    MODELS: Pretrained
        #########################################################
        vgg        = model.vgg16(pretrained=True)
        inception  = model.inception_v3(pretrained=True)
        resnet     = model.resnet50(pretrained=True)
        densenet   = model.densenet121(pretrained=True)
        squeezenet = model.squeezenet1_0(pretrained=True)
        shufflenet = model.shufflenet_v2_x1_0(pretrained=True)
        mobilenet  = model.mobilenet_v2(pretrained=True)
        #########################################################
        #    Robust Models
        #########################################################
        ds = ImageNet('')
        robust_l2_3_0,_ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds
                                             ,resume_path=os.path.join(current_path,'weights/imagenet_l2_eps_3.pt'))
        robust_linf_8,_ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds
                                         ,resume_path=os.path.join(current_path,'weights/imagenet_linf_8.pt'))
        robust_linf_4,_ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds
                                         ,resume_path=os.path.join(current_path,'weights/imagenet_linf_4.pt'))

        # Fast Adversarial Models
        fast_2px = torch.nn.DataParallel(model.resnet50(False))
        fast_4px = torch.nn.DataParallel(model.resnet50(False))
        fast_2px.load_state_dict(torch.load(os.path.join(current_path,'weights/imagenet_2px.pt'))['state_dict'])
        fast_4px.load_state_dict(torch.load(os.path.join(current_path,'weights/imagenet_4px.pt'))['state_dict'])

        self.networks = {
            'vgg': vgg,
            'inception': inception,
            'resnet': resnet,
            'densenet': densenet,
            'squeezenet': squeezenet,
            'mobilenet': mobilenet,
            'shufflenet': shufflenet,
            'robust_l2_3_0': robust_l2_3_0,
            'robust_linf_8': robust_linf_8,
            'robust_linf_4': robust_linf_4,
            'fast_2px': fast_2px,
            'fast_4px': fast_4px
        }

        #########################################################
        #    NORMALIZE
        #########################################################
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = NormalizeByChannelMeanStd(mean, std)


    def get_model(self, model_name):
        return nn.Sequential(self.normalize, self.networks[model_name].eval()).to(self.device)

    def _get_renderers(self):
        # Cameras
        dist, elev_start, elev_end, azim_start, azim_end = self.inter_cam
        elev = torch.linspace(elev_start, elev_end, self.n_views).to(self.device)
        azim = torch.linspace(azim_start, azim_end, self.n_views).to(self.device)
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R.to(self.device), T=T.to(self.device)).to(self.device)
        # Differentiable soft renderer using per vertex RGB colors for texture

        sigma = 1e-4
        raster_settings_soft = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings_soft
            ),
            shader=SoftPhongShader(device=self.device,
                cameras=cameras,
                lights=self.lights)
        )

        return renderer, R,T

    def texturize(self, init_color, path_current, obj_filename, target, n_iter=500, lr=1e-2, bg=None, show_every=100):

        #torch.autograd.set_detect_anomaly(True)
        target_batch =  torch.LongTensor([target]*self.n_views).to(self.device)

        # Load Object
        # Initialize Sphere (Source Mesh)
        src_mesh = load_objs_as_meshes([obj_filename], device=self.device)

        renderer, R, T = self._get_renderers()
        target_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        #########################################################
        #    MULTIVIEW OPTIMIZATION
        #########################################################
        images_predicted = renderer(new_src_mesh.extend(self.n_views), cameras=target_cameras, lights=self.lights)
        # image from our dataset
        predicted_rgb = images_predicted[..., :3].permute(0,3,1,2)
        img_size = (self.image_size, self.image_size)


        with torch.no_grad():
            if i == 1:
                for i,image in enumerate(images_predicted.detach().cpu().numpy()[...,:3]):
                    im = Image.fromarray((image * 255).astype(np.uint8))
                    im.save(f"{path_current}/original_{i}.png")

        return src_mesh


def run_texturization(settings, bg=None):
    lr = settings['lr']
    out_path = settings['out_path']
    n_iter = settings['n_iter']
    device = settings['device']
    n_views = settings['n_views']
    image_size = settings['image_size']

    #Loop Meshes
    for mesh_name, params in settings['meshes'].items():
        # Loop Classifiers
        for ensemble_name, model_names in settings['models'].items():
            # Interpolation Camera settings
            inter_cam = settings['meshes'][mesh_name]['inter_cam']
            target = settings['meshes'][mesh_name]['target']
            init_color = settings['meshes'][mesh_name]['init_color']

            # Camera settings
            dist, elev_start, elev_end, azim_start, azim_end = inter_cam
            elev = torch.linspace(elev_start, elev_end, n_views).to(device)
            azim = torch.linspace(azim_start, azim_end, n_views).to(device)
            R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
            cameras = FoVPerspectiveCameras(device=device, R=R.to(device), T=T.to(device)).to(device)
            # Differentiable soft renderer using per vertex RGB colors for texture
            sigma = 1e-4
            raster_settings_soft = RasterizationSettings(
                image_size=image_size,
                blur_radius=0.0,
                faces_per_pixel=1,
            )

            # Place a point light in front of the object. As mentioned above, the front of
            # the cow is facing the -z direction.
            lights = PointLights(device=device, location=[[10.0, 10.0,10.0]])

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings_soft
                ),
                shader=SoftPhongShader(device=device,
                    cameras=cameras,
                    lights=lights))


            print('*'*50)
            print(f'{mesh_name} - {ensemble_name}')
            print('*'*50)
            print('Loading models...')
            with HiddenPrints():
                texture = GradientTexturization(model_names, n_views, inter_cam, lights)

            path = f'{current_path}/{out_path}/{mesh_name}/'
            # Create Folder
            create_folder(path)
            subfolder = f'{current_path}/{out_path}/{mesh_name}/{ensemble_name}'
            # Create Folder
            create_folder(subfolder)
            obj_filename = f'{os.getcwd()}/meshes/{mesh_name}/{mesh_name}.obj'
            output_mesh = texture.texturize(init_color, path, obj_filename, target, n_iter, lr, bg)
            output_meshes = output_mesh.extend(n_views)
            images_predicted = renderer(output_meshes, cameras=cameras, lights=lights)
            image_grid(images_predicted.cpu().detach().numpy(), rows=1, cols=n_views, rgb=True)
            plt.show()
            # Save Images
            for i,image in enumerate(images_predicted.detach().cpu().numpy()[...,:3]):
                im = Image.fromarray((image * 255).astype(np.uint8))
                im.save(f"{subfolder}/{i}.png")
