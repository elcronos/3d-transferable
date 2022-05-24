#!/usr/bin/env python
# coding: utf-8
import os
import torch
from Texturization import GradientTexturization, run_texturization
import warnings
warnings.filterwarnings("ignore")

settings = {
    'out_path': 'outputs',
    'n_views': 5,
    'n_iter': 60,
    'lr': 5e-2,
    'tv_loss':1e-6,
    'image_size': 512,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'meshes': { 
        'tank':{
           'init_color': [238.,133.,28.], # Initial RGB color
            'inter_cam': [2.0, 0, 20,0,0],
            'target': 366, # Gorilla
        }
    },
    # Different Ensembles
    'models': {
        '1NR':['resnet'],
        '1R':['robust_l2_3_0'],
        '1R&1NR':['robust_l2_3_0','resnet'],
        '2NR':['resnet','densenet'],
        '2R':['robust_l2_3_0','robust_linf_4'],
        '2NR&2R':['resnet','densenet','robust_l2_3_0','robust_linf_4'],
        '4R':['fast_2px','robust_l2_3_0','robust_linf_4','robust_linf_8'],
        '4NR':['resnet','vgg','densenet','inception'],
        'all': ['fast_2px','fast_4px','robust_l2_3_0','robust_linf_4','robust_linf_8','densenet','inception','resnet','vgg'],
    }
}

# Run Texturization
run_texturization(settings)




