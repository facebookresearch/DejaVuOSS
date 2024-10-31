# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import torch

import numpy as np

class stopwatch: 
    def __init__(self, length): 
        self.length = length 
        
    def start(self): 
        self.t0 = time.time()
        
    def elapsed(self): 
        return time.time() - self.t0
    
    def time_remaining(self, i): 
        if i > 0: 
            t_elapse = self.elapsed()
            frac_done = i / self.length
            time_rem = (t_elapse / frac_done) * (1 - frac_done)
        else: 
            time_rem = np.inf
        return time_rem

def most_conf_frac(scores, frac): 
    """get topk NN predictions on the most confident fraction
    of attacked examples. Run after compute_topk_preds
    Input:
        most_conf_frac: scalar [0,1], most confident frac of examps
    Return: 
        frac_idxs: indices of the most confident examples
        preds: topk predictions of these examples 
    """
    n_most_conf = int(frac * len(scores))
        
    #get most confident subset of indices
    most_conf_idxs = np.argsort(scores)[::-1][:n_most_conf]

    #get predictions 
    most_conf_preds = scores[most_conf_idxs]

    return most_conf_idxs, most_conf_preds

def load_vicreg_proj_weights(resnet50_vicreg_intern, ckpt):

    def exclude_bias_and_norm(p):
        return p.ndim == 1

    # TODO fix update buffer
    def update_buffer(module_, k, v):
            A = getattr(module_, k)
            setattr(module_, k, v)

    #print(list(ckpt['model'])[0:2])
    proj_0 = {key[len('module.'):]: ckpt['model'][key] for key in list(ckpt['model'])[0:2]}
    proj_1 = {key[len('module.'):]: ckpt['model'][key] for key in list(ckpt['model'])[2:7]}
    proj_2 = {key[len('module.'):]: ckpt['model'][key] for key in list(ckpt['model'])[7:9]}
    proj_3 = {key[len('module.'):]: ckpt['model'][key] for key in list(ckpt['model'])[9:14]}
    proj_4 = {key[len('module.'):]: ckpt['model'][key] for key in list(ckpt['model'])[14:16]}

    with torch.no_grad():
        # (0)(0): Linear(in_features=2048, out_features=8192, bias=True)
        resnet50_vicreg_intern.module.projector[0][0].weight = torch.nn.Parameter(proj_0['projector.0.weight'])
        resnet50_vicreg_intern.module.projector[0][0].bias = torch.nn.Parameter(proj_0['projector.0.bias'])

        # (0)(1): BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        resnet50_vicreg_intern.module.projector[0][1].weight = torch.nn.Parameter(proj_1['projector.1.weight'])
        resnet50_vicreg_intern.module.projector[0][1].bias = torch.nn.Parameter(proj_1['projector.1.bias'])

        update_buffer(resnet50_vicreg_intern.module.projector[0][1], "running_mean", proj_1['projector.1.running_mean'])
        update_buffer(resnet50_vicreg_intern.module.projector[0][1], "running_var", proj_1['projector.1.running_var'])
        update_buffer(resnet50_vicreg_intern.module.projector[0][1], "num_batches_tracked", proj_1['projector.1.num_batches_tracked'])
        resnet50_vicreg_intern.module.projector[0][1].running_mean = proj_1['projector.1.running_mean']
        resnet50_vicreg_intern.module.projector[0][1].running_var = proj_1['projector.1.running_var']
        resnet50_vicreg_intern.module.projector[0][1].num_batches_tracked = proj_1['projector.1.num_batches_tracked']


        # (1)(0): Linear(in_features=8192, out_features=8192, bias=True)
        resnet50_vicreg_intern.module.projector[1][0].weight = torch.nn.Parameter(proj_2['projector.3.weight'])
        resnet50_vicreg_intern.module.projector[1][0].bias = torch.nn.Parameter(proj_2['projector.3.bias'])

        # (1)(1): BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        resnet50_vicreg_intern.module.projector[1][1].weight = torch.nn.Parameter(proj_3['projector.4.weight'])
        resnet50_vicreg_intern.module.projector[1][1].bias = torch.nn.Parameter(proj_3['projector.4.bias'])

        update_buffer(resnet50_vicreg_intern.module.projector[1][1], "running_mean", proj_3['projector.4.running_mean'])
        update_buffer(resnet50_vicreg_intern.module.projector[1][1], "running_var", proj_3['projector.4.running_var'])
        update_buffer(resnet50_vicreg_intern.module.projector[1][1], "num_batches_tracked", proj_3['projector.4.num_batches_tracked'])
        resnet50_vicreg_intern.module.projector[1][1].running_mean = proj_3['projector.4.running_mean']
        resnet50_vicreg_intern.module.projector[1][1].running_var = proj_3['projector.4.running_var']
        resnet50_vicreg_intern.module.projector[1][1].num_batches_tracked = proj_3['projector.4.num_batches_tracked']

        # (2)(0): Linear(in_features=8192, out_features=8192, bias=False)
        resnet50_vicreg_intern.module.projector[2].weight = torch.nn.Parameter(proj_4['projector.6.weight'])

        optim = ckpt['optimizer']
        
        return optim

def load_barlow_proj_weights(resnet50_intern):

    def exclude_bias_and_norm(p):
        return p.ndim == 1

    barlow_twins_path = 'barlowtwins/checkpoint.pth'
    ckpt_loaded = torch.load(barlow_twins_path, map_location='cuda')

    # load Barlow specific batchnorm and projections
    with torch.no_grad():
        # Load projections
        resnet50_intern.module.projector[0][0].weight.copy_(ckpt_loaded['model']['module.projector.0.weight'])
        resnet50_intern.module.projector[0][1].weight.copy_(ckpt_loaded['model']['module.projector.1.weight'])
        resnet50_intern.module.projector[0][1].bias.copy_(ckpt_loaded['model']['module.projector.1.bias'])
        resnet50_intern.module.projector[0][1].running_mean.copy_(ckpt_loaded['model']['module.projector.1.running_mean'])
        resnet50_intern.module.projector[0][1].running_var.copy_(ckpt_loaded['model']['module.projector.1.running_var'])
        resnet50_intern.module.projector[0][1].num_batches_tracked.copy_(ckpt_loaded['model']['module.projector.1.num_batches_tracked'])

        resnet50_intern.module.projector[1][0].weight.copy_(ckpt_loaded['model']['module.projector.3.weight'])
        resnet50_intern.module.projector[1][1].weight.copy_(ckpt_loaded['model']['module.projector.4.weight'])
        resnet50_intern.module.projector[1][1].bias.copy_(ckpt_loaded['model']['module.projector.4.bias'])
        resnet50_intern.module.projector[1][1].running_mean.copy_(ckpt_loaded['model']['module.projector.4.running_mean'])
        resnet50_intern.module.projector[1][1].running_var.copy_(ckpt_loaded['model']['module.projector.4.running_var'])
        resnet50_intern.module.projector[1][1].num_batches_tracked.copy_(ckpt_loaded['model']['module.projector.4.num_batches_tracked'])

        resnet50_intern.module.projector[2].weight.copy_(ckpt_loaded['model']['module.projector.6.weight'])

        optim = ckpt_loaded['optimizer']
        
        return optim

def load_dino_proj_weights(resnet50_intern, ckpt_loaded):
    # load dino specific batchnorm and projections
    with torch.no_grad():
        print('module.projector.mlp.1.weight: ', ckpt_loaded['student']['module.projector.mlp.1.weight'].shape)
        print('module.projector.mlp.3.weight: ',ckpt_loaded['student']['module.projector.mlp.3.weight'].shape)
        print('last_layer: ', ckpt_loaded['student']['module.projector.last_layer.weight'].shape)
        # Load projections
        resnet50_intern.module.projector[0][0].weight.copy_(ckpt_loaded['student']['module.projector.mlp.0.weight'])
        resnet50_intern.module.projector[0][0].bias.copy_(ckpt_loaded['student']['module.projector.mlp.0.bias'])
        
        resnet50_intern.module.projector[0][1].weight.copy_(ckpt_loaded['student']['module.projector.mlp.1.weight'])
        resnet50_intern.module.projector[0][1].bias.copy_(ckpt_loaded['student']['module.projector.mlp.1.bias'])
        resnet50_intern.module.projector[0][1].running_mean.copy_(ckpt_loaded['student']['module.projector.mlp.1.running_mean'])
        resnet50_intern.module.projector[0][1].running_var.copy_(ckpt_loaded['student']['module.projector.mlp.1.running_var'])
        resnet50_intern.module.projector[0][1].num_batches_tracked.copy_(ckpt_loaded['student']['module.projector.mlp.1.num_batches_tracked'])

        resnet50_intern.module.projector[1].weight.copy_(ckpt_loaded['student']['module.projector.mlp.3.weight'])
        resnet50_intern.module.projector[1].bias.copy_(ckpt_loaded['student']['module.projector.mlp.3.bias'])