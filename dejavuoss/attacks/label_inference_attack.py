# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import faiss
import numpy as np

from dejavuoob.utils.common import stopwatch 
from typing import Callable
from torch.utils.data import DataLoader
from scipy.stats import entropy

class Adversary:

    def __init__(self, model):
        pass

    def setup(self, public_data: DataLoader):
        pass

    def attack(self, data: DataLoader):
        pass
    
    def get_latest_results(self, top_k: int, most_conf_frac: int):
        pass
 
    def get_most_conf_fraction(self, topk_preds: torch.Tensor, most_conf_frac: int):
        pass
 
def default_get_emb_fn(model: torch.nn.Module, use_backbone: bool = True, use_supervised_linear: bool = False):
    '''
    TODO: make this more generic. 
    '''
    x = x.cuda()
    embed =model(x)
    if not use_backbone:
        embed = model.module.projector(embed)
        if use_supervised_linear: 
            embed = model.module.fc(embed)
    return embed


class KNNAdversary(Adversary):

    def __init__(self, model: torch.nn.Module, public_data: DataLoader, 
                 get_emb_fn: Callable = default_get_emb_fn, k: int = 100):
        self.model = model
        self.public_data = public_data
        self.get_emb_fn = get_emb_fn
        self.k = k

        # initializing instance variables
        self.public_labels = []
        self.public_idxs = []

    def setup(self, progress_interval: int = 10, device: int = 1):
        self._build_index(progress_interval, device)

    def _build_index(self, progress_interval: int = 10, device: int = 1): 
        '''
        Always runs on GPU ... 
        '''
        public_inputs = []
        n = len(self.public_data)

        interval = n // progress_interval

        sw = stopwatch(n)
        sw.start()

        print('gathering public embeddings...')
        for i, (x,y,idx) in enumerate(self.public_data): 
            # TODO this should happen in the get_emb_fn
            with torch.no_grad(): 
                embed = self.get_emb_fn(x)

                public_inputs.append(embed.cpu().numpy())
                self.public_labels.append(y.numpy())
                self.public_idxs.append(idx.numpy())
            if (i+1) % interval == 0: 
                print(f"progress: {i/n:.2f}, min remaining: {sw.time_remaining(i)/60:.1f}")

        public_inputs = np.concatenate(public_inputs, axis = 0)
        self.public_labels = np.concatenate(self.public_labels, axis = 0)
        self.public_idxs = np.concatenate(self.public_idxs, axis = 0)
        
        print("Building faiss index ...")
        quantizer = faiss.IndexFlatL2(public_inputs.shape[1])
        res = faiss.StandardGpuResources()

        self.fais_index = faiss.index_cpu_to_gpu(res, device, quantizer)
        self.fais_index.train(public_inputs)
        self.fais_index.add(public_inputs)

    def attack(self, data: DataLoader, progress_interval: int = 10):
        n = len(data)
        interval = n // progress_interval

        sw = stopwatch(n)
        sw.start()

        print("getting neighbors...")
        for i, (x,y,idx) in enumerate(data): 
            with torch.no_grad():
                embed = self.get_emb_fn(x)
            
            #get idxs with usable bounding boxes 
            good_idx = y>-1
            embed = embed[good_idx].cpu().numpy()
            y = y[good_idx].numpy()
            idx = idx[good_idx]
            
            if len(y) > 0: 
                #get indices of nearest neighbors
                D,I = self.index.search(embed, self.k) 
                self.neighb_idxs.append(self.public_idxs[I])

                #get labels of nearest neighbors 
                k_neighb_labels = self.public_labels[I.ravel()]
                k_neighb_labels = k_neighb_labels.reshape(I.shape)
                self.neighb_labels.append(k_neighb_labels)
                
                #get indices of the examples attacked 
                #(that we just got neighbors of)
                self.attk_idxs.append(idx)

            if (i+1) % interval == 0: 
                print(f"progress: {i/n:.2f}, min remaining: {sw.time_remaining(i)/60:.1f}")

        self.neighb_idxs = np.concatenate(self.neighb_idxs)
        self.neighb_labels = np.concatenate(self.neighb_labels)
        self.attk_idxs = np.concatenate(self.attk_idxs)

        #get class counts 
        self.class_cts = np.apply_along_axis(np.bincount, axis=1, 
                                arr=self.neighb_labels[:,:self.k_attk], minlength=1000)

        #get confidence 
        self.attk_uncert = entropy(self.class_cts, axis = 1)

    def get_latest_results(self, top_k: int):
        """get topk NN predictions on all attacked examples
        Input:
            top_k: compute top k NN predictions 
        """
        _, topk_preds = torch.topk(torch.Tensor(self.class_cts), top_k, dim = 1)
        topk_preds = np.array(topk_preds).astype(int)

        return topk_preds

    def get_most_conf_fraction(self, topk_preds: torch.Tensor, most_conf_frac: int):
        n_most_conf = int(most_conf_frac * len(self.attk_uncert))
        
        #get most confident subset of indices
        most_conf_idxs = np.argsort(self.attk_uncert)[:n_most_conf]
        
        #get predictions 
        most_conf_preds = topk_preds[most_conf_idxs, :]
        
        return self.attk_idxs[most_conf_idxs], most_conf_preds
       
