# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torchmetrics
import numpy as np
import argparse
from pathlib import Path
from utils.models import SSLNetwork
from utils.image_common import ImageFolderWithPath
from torch.utils.data import DataLoader
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
import ffcv
from ffcv.loader import OrderOption
from ffcv.fields.basics import IntDecoder
from torch.cuda.amp import autocast
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from scipy.stats import entropy

class SSL_Transform2: 
    """Transform applied to SSL examples at test time 
    """
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.ssl_xfrm = transforms.Compose([
                    transforms.Resize((224, 224)),
                    #transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                        ])
        
    def __call__(self, x): 
       return self.ssl_xfrm(x)

def gpu_init(gpu):
    if not torch.distributed.is_initialized(): 
        dist_url = Path(os.path.join('/scratch/', 'interactive_init'))
        if dist_url.exists():
            os.remove(str(dist_url))
        dist_url = dist_url.as_uri()

        torch.distributed.init_process_group(
            backend='nccl', init_method=dist_url,
            world_size=1, rank=0)                                    

    torch.cuda.set_device(gpu) 

def load_model(args):
    if args.resnet50: 
        arch = 'resnet50'
    else:
        arch = 'resnet101'

    model_path = args.model_path

    gpu_init(args.gpu)

    resnet_crop = SSLNetwork(arch = arch,
                             loss = args.loss).cuda()
    resnet_crop = torch.nn.parallel.DistributedDataParallel(resnet_crop, device_ids=[args.gpu])
    ckpt = torch.load(model_path, map_location='cpu')

    resnet_crop.load_state_dict(ckpt['model'], strict = True)

    resnet_crop.to(memory_format=torch.channels_last)

    resnet_crop.eval()

    return resnet_crop

def create_val_loader(val_dataset, num_workers, batch_size,
                          resolution, distributed, gpu):
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
        IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
        DEFAULT_CROP_RATIO = 224/256
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(torch.device(gpu), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(gpu),
            non_blocking=True)
        ]

        order = OrderOption.SEQUENTIAL

        loader = ffcv.Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

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

def evaluate(args):
    # load model
    model = load_model(args)

    # load dataset
    dataset = ImageFolderWithPath(args.crop_dir, SSL_Transform2())
    dataloader = DataLoader(dataset, batch_size = 64, shuffle = False, num_workers=8)

    # iterate over dataset and make predictions
    targets = []
    predictions = []
    scores = []
    softmaxes = []
    paths = []
    y_scores = []
    entropies = []
    for x, y, path in dataloader: 
        x = x.cuda()#.half()
        y = y.cuda()

        with torch.no_grad():
            with autocast():
                output = model(x)
                softmax = output.softmax(dim=-1).detach()
                score, index = torch.max(softmax, dim = -1)
                entrpy = entropy(softmax.detach().cpu().numpy(), axis=-1)
                y_score = softmax.gather(1, y.unsqueeze(1))

                prediction = index

                targets.append(y)
                predictions.append(prediction)
                scores.append(score)
                softmaxes.append(softmax)
                paths.append(path)
                y_scores.append(y_score.squeeze(-1))
                entropies.append(entrpy)

    targets = torch.cat(targets)
    predictions = torch.cat(predictions)
    scores = torch.cat(scores)
    y_scores = torch.cat(y_scores)
    entropies = np.concatenate(entropies)
    softmaxes = torch.cat(softmaxes)
    paths = np.concatenate(paths)

    accs_torchmetrics = torchmetrics.Accuracy('multiclass',num_classes=1000,top_k=1).to(args.gpu)
    mcss_torchmetrics = torchmetrics.classification.MulticlassStatScores(num_classes=1000, average=None).to(args.gpu)

    tp = mcss_torchmetrics(softmaxes, targets)[:,0]
    tp_indices = torch.nonzero(tp, as_tuple=True)[0]
    tp_values = tp[tp_indices]

    #print('indices equal ? : ', indices == indices_from_dl)
    print('accs_torchmetrics: ', accs_torchmetrics(softmaxes, targets))
    print('mcss_torchmetrics tp_indices: ', tp_indices.cpu().numpy().tolist())
    print('mcss_torchmetrics tp_values: ', tp_values.cpu().numpy().tolist())

    return None, targets, predictions, scores, y_scores, entropies, paths


def parse_args(): 
    parser = argparse.ArgumentParser("plotting args")
    parser.add_argument("--model_path", type=Path,
            help="path of the model to perform evaluation on")
    parser.add_argument("--gpu", default=0, type=int, 
            help="gpu device")
    parser.add_argument("--crop_dir", type=Path,
            help="crop dir path")
    parser.add_argument("--resnet50", action='store_true')
    parser.add_argument("--loss", type=str, default='vicreg')
    parser.add_argument("--save_path", type=Path, default='/',
            help="save evaluation results")
    return parser.parse_args()

def compute_accuracy(accs_targets, predictions, accs_indices_subset):
    return (predictions[accs_indices_subset] == accs_targets[accs_indices_subset]).mean()

def main(args): 
    print('Evaluating crop training results ...')
    
    # create saving path
    #Path(args.save_path).mkdir(parents=True, exist_ok=True)

    indices_from_dl, targets, predictions, scores, y_scores, entropies, paths  = evaluate(args)

    #indices_from_dl = indices_from_dl.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    y_scores = y_scores.detach().cpu().numpy()
    print('y_scores: ', y_scores.shape)

    score_indices_subset, _ = most_conf_frac(scores, 1)
    print('Crop model Acc on Test: ', compute_accuracy(targets, predictions, score_indices_subset))

    score_indices_subset, _ = most_conf_frac(scores, 0.05)
    print('Crop model Acc on Test 5%: ', compute_accuracy(targets, predictions, score_indices_subset))

    score_indices_subset, _ = most_conf_frac(scores, 0.2)
    print('Crop model Acc on Test 20%: ', compute_accuracy(targets, predictions, score_indices_subset))

    # stack the results in one array and save
    eval = np.stack((targets, predictions, scores, y_scores, entropies, paths), axis = -1)
    np.save(args.save_path, eval)


if __name__ == "__main__":
    args = parse_args()

    main(args)