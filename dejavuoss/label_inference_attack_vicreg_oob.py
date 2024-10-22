# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#modified model for gen ssl trained files

import numpy as np
import torch, torchvision
from torchvision import transforms
from torch import nn, optim
from torchvision.datasets import ImageFolder
from torchvision import datasets
from torchvision.models import resnet50, resnet101
from torch.utils.data import DataLoader, Subset
from matplotlib import pyplot as plt
import argparse
import os, sys
from pathlib import Path
import time
import xmltodict
import faiss
from scipy.stats import entropy
import submitit
import uuid
from itertools import groupby, chain
from operator import itemgetter

#for loading datasets: 
from dejavu_utils.utils import (aux_dataset, crop_dataset, 
                                ImageFolderIndex, stopwatch, 
                                SSLNetwork, SSL_Transform, load_vicreg_proj_weights)

def parse_args():
    parser = argparse.ArgumentParser("Submitit for NN Attack")

    parser.add_argument("--local", default = 0, type=int, help="whether to run on devfair")
    parser.add_argument("--local_gpu", default = 1, type=int, help="which device to use during local run")
    #slurm args 
    parser.add_argument("--timeout", default=2800, type=int, help="Duration of the job")
    parser.add_argument("--partition", default="", type=str, help="Partition where to submit")
    parser.add_argument("--mem_gb", default=250) 
    parser.add_argument("--output_dir", type=Path) 

    #attack args
    parser.add_argument("--model_A_pth", type=Path) 
    parser.add_argument("--mlp", type=str, default='8192-8192-8192') 
    parser.add_argument("--use_backbone", type=int, default=0) 
    parser.add_argument("--loss", type=str, default='barlow') 
    parser.add_argument("--use_supervised_linear", type=int, default=0) 
    parser.add_argument("--use_corner_crop", action='store_true')
    parser.add_argument("--corner_crop_frac", type=float, default=0.3)
    parser.add_argument("--public_idx_pth", type=Path) 
    parser.add_argument("--test_idx_pth", type=Path) 
    parser.add_argument("--valid_idx_pth", type=Path)
    parser.add_argument("--imgnet_train_pth", type=Path, default="")
    parser.add_argument("--imgnet_valid_pth", type=Path, default="")
    parser.add_argument("--imgnet_bbox_pth", type=Path, default="") 
    parser.add_argument("--imgnet_valid_bbox_pth", type=Path, default="") 
    parser.add_argument("--k", type=int, default=100, 
            help="number of neighbors to search when building index") 
    parser.add_argument("--k_attk", type=int, default=100, 
            help="number of neighbors to use in attack") 
    parser.add_argument("--resnet50", action='store_true')
             
    
    return parser.parse_args()

#NN_adversary(model_A, public_loader, args.gpu, args.k, args.k_attk, args.use_backbone)
class NN_adversary: 
    def __init__(self, model, public_DL, args):# gpu, k = 100, k_attk = None, use_backbone = 0): 
        self.model = model
        self.use_backbone = args.use_backbone
        self.public_DL = public_DL
        self.use_supervised_linear = args.use_supervised_linear and (args.loss == 'supervised')

        self.gpu = args.gpu

        #Nearest neighbor index 
        self.k = args.k #number of neighbors to collect, not necessarily use in attack 
        self.index = None
        self.public_idxs = []
        self.public_labels = []
        
        #Nearest neighbor data on attk set 
        self.neighb_idxs = [] 
        self.neighb_labels = []
        self.attk_idxs = []
        self.class_cts = []
        
        #activation attack 
        self.activations = []
        
        #attack uncertainty on attk set 
        self.attk_uncert = []
        self.topk_preds = []
        
        #num neighbs 
        if not args.k_attk: 
            self.k_attk = self.k
        else: 
            self.k_attk = args.k_attk

    def get_embed(self, x): 
        embed = self.model.module.net(x)
        if not self.use_backbone:
            embed = self.model.module.projector(embed)
            if self.use_supervised_linear: 
                embed = self.model.module.fc(embed)
        return embed

        
    def build_index(self): 
        DS = []
        n = len(self.public_DL)
        print_every = int(n / 10)
        sw = stopwatch(n)
        sw.start()
        print('gathering public embeddings...')
        for i, (x,y,idx) in enumerate(self.public_DL): 
            x = x.cuda()
            with torch.no_grad(): 
                embed = self.get_embed(x)

                DS.append(embed.cpu().numpy())
                self.public_labels.append(y.numpy())
                self.public_idxs.append(idx.numpy())
            if (i+1) % print_every == 0: 
                print(f"progress: {i/n:.2f}, min remaining: {sw.time_remaining(i)/60:.1f}")

        DS = np.concatenate(DS, axis = 0)
        self.public_labels = np.concatenate(self.public_labels, axis = 0)
        self.public_idxs = np.concatenate(self.public_idxs, axis = 0)
        
        print("building faiss index...")
        nlist = 1000 #number of voronoi cells in NN indexer
        dim = DS.shape[1]
        quantizer = faiss.IndexFlatL2(dim)
        
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, self.gpu, quantizer)
        
        self.index.train(DS)
        self.index.add(DS)
        
    def get_neighbors(self, aux_DL): 
        n = len(aux_DL)
        print_every = int(n / 10)
        sw = stopwatch(n)
        sw.start()
        print("getting neighbors...")
        for i, (x,y,idx) in enumerate(aux_DL): 
            '''
            if x == -1:
                print(f'No annotation found for the idx: {idx}')
                continue
            '''
            with torch.no_grad():
                embed = self.get_embed(x.cuda())

            #get idxs with usable bounding boxes 
            good_idx = y>-1
            embed = embed[good_idx].cpu().numpy()
            y = y[good_idx].numpy()
            #print('bad idx: ', idx[~good_idx])

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
                
            if (i+1) % print_every == 0: 
                print(f"progress: {i/n:.2f}, min remaining: {sw.time_remaining(i)/60:.1f}")
                
        self.neighb_idxs = np.concatenate(self.neighb_idxs)
        self.neighb_labels = np.concatenate(self.neighb_labels)
        self.attk_idxs = np.concatenate(self.attk_idxs)
        
        #get class counts 
        self.class_cts = np.apply_along_axis(np.bincount, axis=1, 
                                arr=self.neighb_labels[:,:self.k_attk], minlength=1000)
        
        #get confidence 
        self.attk_uncert = entropy(self.class_cts, axis = 1)
        
    def get_activations(self, aux_DL): 
        n = len(aux_DL)
        print_every = int(n / 10)
        sw = stopwatch(n)
        sw.start()
        print("getting neighbors...")
        for i, (x,y,idx) in enumerate(aux_DL): 
            '''
            if x == -1:
                print(f'No annotation found for the idx: {idx}')
                continue
            '''
            with torch.no_grad():
                embed = self.get_embed(x.cuda())
            
            #get idxs with usable bounding boxes 
            good_idx = y>-1
            embed = embed[good_idx].cpu().numpy()
            y = y[good_idx].numpy()
            idx = idx[good_idx]
            
            if len(y) > 0: 
                self.activations.append(embed)
                self.attk_idxs.append(idx)

            if (i+1) % print_every == 0: 
                print(f"progress: {i/n:.2f}, min remaining: {sw.time_remaining(i)/60:.1f}")
                
        self.activations = np.concatenate(self.activations)
        self.attk_idxs = np.concatenate(self.attk_idxs)
        
        #get class counts 
        self.class_cts = self.activations
        
        #get confidence 
        self.attk_uncert = - np.max(self.class_cts, axis = 1)
        
    def compute_topk_preds(self, k): 
        """get topk NN predictions on all attacked examples
        Input:
            k: compute top k NN predictions 
        """
        topk_cts, topk_preds = torch.topk(torch.Tensor(self.class_cts), k, dim = 1)
        self.topk_preds = np.array(topk_preds).astype(int)
        
    def attack_p_frac(self, most_conf_frac): 
        """get topk NN predictions on the most confident fraction
        of attacked examples. Run after compute_topk_preds
        Input:
            most_conf_frac: scalar [0,1], most confident frac of examps
        Return: 
            frac_idxs: indices of the most confident examples
            preds: topk predictions of these examples 
        """
        n_most_conf = int(most_conf_frac * len(self.attk_uncert))
        
        #get most confident subset of indices
        most_conf_idxs = np.argsort(self.attk_uncert)[:n_most_conf]

        #get predictions 
        most_conf_preds = self.topk_preds[most_conf_idxs, :]
        
        return self.attk_idxs[most_conf_idxs], most_conf_preds

def exclude_bias_and_norm(p):
    return p.ndim == 1

#Run attack code
def main(args): 
    #init distributed process because saved models need this
    print('Initializing process group...') 
    torch.distributed.init_process_group(
       backend='nccl', init_method=args.dist_url,
       world_size=args.world_size, rank=args.rank)
   
    torch.cuda.set_device(args.gpu)

    print('Loading models...')

    #load up models A and B 
    if args.resnet50: 
        arch = 'resnet50'
    else:
        arch = 'resnet101'

    resnet50_vicreg_intern = SSLNetwork(arch = arch, 
                                        remove_head = 0, 
                                        mlp = args.mlp, 
                                        fc = 0,
                                        patch_keep = 1.0,
                                        loss = args.loss).cuda()

    resnet50_vicreg_intern = torch.nn.parallel.DistributedDataParallel(resnet50_vicreg_intern,
                                                                       device_ids=[args.gpu])
 
    ckpt = torch.load(args.model_A_pth, map_location='cpu')
    resnet50_vicreg_intern.load_state_dict(ckpt['model'], strict = True)
    _ = resnet50_vicreg_intern.eval()

    #Get datasets
    #bbox set A 
    print('initializing datasets...')    
    knn_idx = np.load(args.public_idx_pth)
    knn_set = ImageFolderIndex(args.imgnet_train_pth,  SSL_Transform(), knn_idx)
    knn_loader = DataLoader(knn_set, batch_size = 64, shuffle = False, num_workers=8)

    #test_set = ImageFolderIndex(args.imgnet_train_pth,  SSL_Transform(), test_idx)
    #test_loader = DataLoader(test_set, batch_size = 64, shuffle = False, num_workers=8)
    test_idx = np.load(args.test_idx_pth)
    if not args.use_corner_crop: 
        aux_set_test = aux_dataset(args.imgnet_train_pth, args.imgnet_bbox_pth, test_idx)
    else: 
        aux_set_test = crop_dataset(args.imgnet_train_pth, test_idx, crop_frac = args.corner_crop_frac)

    aux_loader_test = DataLoader(aux_set_test, batch_size = 64, num_workers = 8, shuffle = True)
    
    valid_idx = np.load(args.valid_idx_pth)
    if not args.use_corner_crop:
        aux_set_valid = aux_dataset(args.imgnet_valid_pth, args.imgnet_valid_bbox_pth, valid_idx)
    else:
        aux_set_valid = crop_dataset(args.imgnet_valid_pth, valid_idx, crop_frac = args.corner_crop_frac)

    aux_loader_valid = DataLoader(aux_set_valid, batch_size = 64, num_workers = 8, shuffle = True)

    #full imgnet dataset 
    train_imgnet_set = ImageFolder(args.imgnet_train_pth, transform = transforms.ToTensor())
    valid_imgnet_set = ImageFolder(args.imgnet_valid_pth, transform = transforms.ToTensor())

    #Now attack test set with adversary A 

    adv_test_attk_A = NN_adversary(resnet50_vicreg_intern, knn_loader, args)
    adv_test_attk_A.build_index()
    adv_test_attk_A.get_neighbors(aux_loader_test)
    #free gpu memory 
    adv_test_attk_A.index.reset()

    #Now attack sets A & B with adversary A 
    adv_valid_attk_A = NN_adversary(resnet50_vicreg_intern, knn_loader, args)
    adv_valid_attk_A.build_index()
    adv_valid_attk_A.get_neighbors(aux_loader_valid)
    #free gpu memory 
    adv_valid_attk_A.index.reset()

    def get_labels_train(idxs): 
        #get ground-truth labels of examples indices
        return np.array([train_imgnet_set.samples[i][1] for i in idxs])[:,None]

    def get_labels_valid(idxs): 
        #get ground-truth labels of examples indices
        return np.array([valid_imgnet_set.samples[i][1] for i in idxs])[:,None]

    #test attk A
    np.save(args.output_dir / 'test_attk_A_neighb_idxs', adv_test_attk_A.neighb_idxs)
    np.save(args.output_dir / 'test_attk_A_neighb_labels', adv_test_attk_A.neighb_labels)
    np.save(args.output_dir / 'test_attk_A_attk_idxs', adv_test_attk_A.attk_idxs)
    np.save(args.output_dir / 'test_attk_A_labels', get_labels_train(adv_test_attk_A.attk_idxs))

    #valid attk A
    np.save(args.output_dir / 'valid_attk_A_neighb_idxs', adv_valid_attk_A.neighb_idxs)
    np.save(args.output_dir / 'valid_attk_A_neighb_labels', adv_valid_attk_A.neighb_labels)
    np.save(args.output_dir / 'valid_attk_A_attk_idxs', adv_valid_attk_A.attk_idxs)
    np.save(args.output_dir / 'valid_attk_A_labels', get_labels_valid(adv_valid_attk_A.attk_idxs))

    def get_acc_train_group_by_label(idxs, preds): 
        #get array indicating whether topk preds are correct
        true_labels = get_labels_train(idxs)
        accs = true_labels == preds
        true_labels_flatten, accs_flatten = np.array(true_labels).flatten(), np.array(accs).flatten()
        true_labels_sorted_idx = np.argsort(true_labels_flatten)
        grouped_accs = [(key, list(group)) for key, group in groupby(zip(true_labels_flatten[true_labels_sorted_idx], \
            accs_flatten[true_labels_sorted_idx]), lambda x: x[0])]
        #print('grouped_accs len: ', len(grouped_accs))
        #print('grouped_accs: ', grouped_accs)

        return {key: sum([e2 for _, e2 in group]) / len(group) for key, group in grouped_accs}

    def get_acc_train(idxs, preds): 
        #get array indicating whether topk preds are correct
        true_labels = get_labels_train(idxs) 
        return (true_labels == preds).sum(axis = 1)    


    def get_acc_valid(idxs, preds): 
        #get array indicating whether topk preds are correct
        true_labels = get_labels_valid(idxs) 
        return (true_labels == preds).sum(axis = 1)    

    def get_acc_valid_group_by_label(idxs, preds): 
        #get array indicating whether topk preds are correct
        true_labels = get_labels_valid(idxs) # flatten labels
        accs = true_labels == preds
        true_labels_flatten, accs_flatten = np.array(true_labels).flatten(), np.array(accs).flatten()
        true_labels_sorted_idx = np.argsort(true_labels_flatten)
        grouped_accs = [(key, list(group)) for key, group in groupby(zip(true_labels_flatten[true_labels_sorted_idx], \
            accs_flatten[true_labels_sorted_idx]), lambda x: x[0])]
        #for key, group in grouped_accs:
        #    print(key, sum([e2 for _, e2 in group]), len(group))
        
        return {key: sum([e2 for _, e2 in group]) / len(group) for key, group in grouped_accs}

    topks = [1,5]

    for topk in topks: 
        print(f"top-{topk} results:")

        adv_test_attk_A.compute_topk_preds(topk)
        adv_valid_attk_A.compute_topk_preds(topk)

        print("Attack stats on set A")

        #get marginal accuracy on set A 
        idxs_test, preds_test = adv_test_attk_A.attack_p_frac(most_conf_frac = 1)
        acc = get_acc_train(idxs_test, preds_test).mean()
        print(f"Model A Acc on Test: {acc:.3f}")
        if topk == 1:
            acc_train_group_by_label = get_acc_train_group_by_label(idxs_test, preds_test)
            np.save(args.output_dir / f"top_{topk}_acc_train_group_by_label", acc_train_group_by_label)

        idxs_valid, preds_valid = adv_valid_attk_A.attack_p_frac(most_conf_frac = 1)
        acc = get_acc_valid(idxs_valid, preds_valid).mean()
        print(f"Model A Acc on Valid: {acc:.3f}")
        if topk == 1:
            acc_valid_group_by_label = get_acc_valid_group_by_label(idxs_valid, preds_valid)
            np.save(args.output_dir / f"top_{topk}_acc_valid_group_by_label", acc_valid_group_by_label)

        #get conf conditioned accuracy on set A 
        idxs_test_p05, preds_test_p05 = adv_test_attk_A.attack_p_frac(most_conf_frac = .05)
        acc = get_acc_train(idxs_test_p05, preds_test_p05).mean()
        print(f"Model A 5% most conf Acc on Test: {acc:.3f}")
        
        idxs_valid_p05, preds_valid_p05 = adv_valid_attk_A.attack_p_frac(most_conf_frac = .05)
        acc = get_acc_valid(idxs_valid_p05, preds_valid_p05).mean()
        print(f"Model A 5% most conf Acc on Valid: {acc:.3f}")

        idxs_test_p20, preds_test_p20 = adv_test_attk_A.attack_p_frac(most_conf_frac = .20)
        acc = get_acc_train(idxs_test_p20, preds_test_p20).mean()
        print(f"Model A 20% most conf Acc on Test: {acc:.3f}")
        
        idxs_valid_p20, preds_valid_p20 = adv_valid_attk_A.attack_p_frac(most_conf_frac = .20)
        acc = get_acc_valid(idxs_valid_p20, preds_valid_p20).mean()
        print(f"Model A 20% most conf Acc Valid: {acc:.3f}")


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        self._setup_gpu_args()
        main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")



def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def file_submitit_job(args): 
    if args.output_dir == "":
        args.output_dir = get_shared_folder() / "%j"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)
    kwargs = {}

    executor.update_parameters(
            mem_gb=args.mem_gb,
            gpus_per_node=1,
            tasks_per_node=1,  # one task per GPU
            cpus_per_task=10,
            nodes=1,
            timeout_min=args.timeout,  # max is 60 * 72
            # Below are cluster dependent parameters
            slurm_partition=args.partition,
            slurm_signal_delay_s=120,
            **kwargs
        )

    executor.update_parameters(name="NN attack")

    args.dist_url = get_init_file().as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    if args.local == 1: 
        if args.output_dir == "":
            args.output_dir = get_shared_folder() / "%j"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.dist_url = get_init_file().as_uri()
        args.gpu = args.local_gpu
        args.world_size = 1
        args.rank = 0
        main(args)       
    else: 
        file_submitit_job(args)
