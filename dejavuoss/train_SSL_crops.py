# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/libffcv/ffcv-imagenet to support SSL
import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)
import torchvision.transforms as transforms
import torchmetrics
import numpy as np
from tqdm import tqdm
import subprocess
import os
import copy
import time
import json
import uuid
import ffcv
import submitit
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser
from torchvision.utils import make_grid, save_image
from torchmetrics.utilities.enums import DataType
from collections import Counter
from fastargs import get_current_config, set_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from dejavu_utils.train_utils import LARS, cosine_scheduler, learning_schedule
from torchmetrics.classification import MulticlassStatScores

Section('model', 'model details').params(
    arch=Param(str, 'model to use', default='resnet50'),
    remove_head=Param(int, 'remove the projector? (1/0)', default=1),
    mlp=Param(str, 'number of projector layers', default="2048-512"),
    mlp_coeff=Param(float, 'number of projector layers', default=1),
    patch_keep=Param(float, 'Proportion of patches to keep with VIT training', default=1.0),
    fc=Param(int, 'remove the projector? (1/0)', default=1),
    proj_relu=Param(int, 'Proj relu? (1/0)', default=0),
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=64),
    max_res=Param(int, 'the maximum (starting) resolution', default=224),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=30),
    start_ramp=Param(int, 'when to start interpolating resolution', default=10)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', default=""),
    num_workers=Param(int, 'The number of workers', default=10),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True),
    random_seed=Param(int, 'Purcentage of noised labels', default=-0),
)

Section('attack', 'attack').params(
    k=Param(int, 'number of neigbors faiss', default=100),
    k_attk=Param(int, 'number of neigbors attack', default=100),
)

Section('vicreg', 'Vicreg').params(
    sim_coeff=Param(float, 'VicREG MSE coefficient', default=25),
    std_coeff=Param(float, 'VicREG STD coefficient', default=25),
    cov_coeff=Param(float, 'VicREG COV coefficient', default=1),
)

Section('simclr', 'simclr').params(
    temperature=Param(float, 'SimCLR temperature', default=0.15),
)

Section('barlow', 'barlow').params(
    lambd=Param(float, 'Barlow Twins Lambd parameters', default=0.0051),
)

Section('byol', 'byol').params(
    momentum_teacher=Param(float, 'Momentum Teacher value', default=0.996),
)

Section('dino', 'dino').params(
    warmup_teacher_temp=Param(float, 'weight decay', default=0.04),
    teacher_temp=Param(float, 'weight decay', default=0.07),
    warmup_teacher_temp_epochs=Param(int, 'weight decay', default=50),
    student_temp=Param(float, 'center momentum dino', default=0.1),
    center_momentum=Param(float, 'center momentum dino', default=0.9),
    momentum_teacher=Param(float, 'Momentum Teacher value', default=0.996),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
    log_level=Param(int, '0 if only at end 1 otherwise', default=2),
    checkpoint_freq=Param(int, 'When saving checkpoints', default=5), 
    snapshot_freq=Param(int, 'How often to save a model snapshot', default=50), 
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=256),
    resolution=Param(int, 'final resized validation image size', default=224),
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    eval_freq=Param(float, 'number of epochs', default=50),
    batch_size=Param(int, 'The batch size', default=512),
    num_small_crops=Param(int, 'number of crops?', default=0),
    optimizer=Param(And(str, OneOf(['sgd', 'adamw', 'lars'])), 'The optimizer', default='adamw'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=30),
    base_lr=Param(float, 'number of epochs', default=0.0005),
    end_lr_ratio=Param(float, 'number of epochs', default=0.001),
    label_smoothing=Param(float, 'label smoothing parameter', default=0),
    distributed=Param(int, 'is distributed?', default=0),
    clip_grad=Param(float, 'sign the weights of last residual block', default=0),
    use_ssl=Param(int, 'use ssl data augmentations?', default=0),
    loss=Param(str, 'use ssl data augmentations?', default="simclr"),
    train_probes_only=Param(int, 'load linear probes?', default=0),
)

Section('dist', 'distributed training options').params(
    use_submitit=Param(int, 'enable submitit', default=0),
    world_size=Param(int, 'number gpus', default=1),
    ngpus=Param(int, 'number of gpus per nodes', default=8),
    nodes=Param(int, 'number of nodes', default=1),
    comment=Param(str, 'comment for slurm', default=''),
    timeout=Param(int, 'timeout', default=2800),
    partition=Param(str, 'partition', default=""),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='58492')
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

################################
##### Some Miscs functions #####
################################

def get_shared_folder() -> Path:
    user = os.getenv("USER")
    path = "/checkpoint/"
    if Path(path).is_dir():
        p = Path(f"{path}{user}/experiments")
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

def exclude_bias_and_norm(p):
    return p.ndim == 1

################################
##### SSL Model Generic CLass ##
################################

class SSLNetwork(nn.Module):
    @param('model.arch')
    #@param('model.remove_head')
    #@param('model.mlp')
    #@param('model.patch_keep')
    #@param('model.fc')
    @param('training.loss')
    def __init__(
        self, arch, loss
    ):
        super().__init__()
        if "resnet" in arch:
            import torchvision.models.resnet as resnet
            self.net = resnet.__dict__[arch]()
        else:
            print("Arch not found")
            exit(0)

        self.num_features = self.net.fc.in_features
        self.loss = loss

    def forward(self, inputs):
        return self.net(inputs)

################################
##### Main Trainer ############
################################

class ImageNetTrainer:
    @param('training.distributed')
    @param('training.batch_size')
    @param('training.label_smoothing')
    @param('training.loss')
    @param('training.train_probes_only')
    @param('training.epochs')
    @param('training.num_small_crops')
    @param('data.train_dataset')
    @param('data.val_dataset')
    def __init__(self, gpu, ngpus_per_node, world_size, dist_url, distributed, batch_size, label_smoothing, loss, train_probes_only, epochs, num_small_crops, train_dataset, val_dataset):
        self.all_params = get_current_config()
        ch.cuda.set_device(gpu)
        self.gpu = gpu
        self.rank = self.gpu + int(os.getenv("SLURM_NODEID", "0")) * ngpus_per_node
        self.world_size = world_size
        self.seed = 50 + self.rank
        self.dist_url = dist_url
        self.batch_size = batch_size
        self.uid = str(uuid4())
        if distributed:
            self.setup_distributed()
        self.start_epoch = 0
        # Create dataLoader used for training and validation
        self.index_labels = 1
        self.train_loader = self.create_train_loader_ssl(train_dataset) #TODO changed for supervised
        self.num_train_exemples = self.train_loader.indices.shape[0]
        self.num_classes = 1000
        self.val_loader = self.create_val_loader(val_dataset)
        self.num_val_exemples = self.val_loader.indices.shape[0]
        print("NUM TRAINING EXEMPLES:", self.num_train_exemples)
        print("NUM VAL EXEMPLES:", self.num_val_exemples)
        # Create SSL model
        self.model, self.scaler = self.create_model_and_scaler()
        self.num_features = self.model.module.num_features

        self.initialize_logger()
        self.classif_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.create_optimizer()

        # Define SSL loss
        self.loss_name = loss
        print('self.loss_name: ', self.loss_name)
 
    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    #@param('training.use_ssl')
    #@param('data.train_dataset')
    def get_dataloader(self):
        return self.create_train_loader_ssl(160), self.create_val_loader()

    def setup_distributed(self):
        dist.init_process_group("nccl", init_method=self.dist_url, rank=self.rank, world_size=self.world_size)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing):
        assert optimizer == 'sgd' or optimizer == 'adamw' or optimizer == "lars"

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]
        if optimizer == 'sgd':
            self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        elif optimizer == 'adamw':
            # We use a big eps value to avoid instabilities with fp16 training
            self.optimizer = ch.optim.AdamW(param_groups, lr=1e-4)
        elif optimizer == "lars":
            self.optimizer = LARS(param_groups)  # to use with convnet and large batches
        #self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optim_name = optimizer

    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('data.in_memory')
    @param('training.num_small_crops')
    def create_train_loader_ssl(self, train_dataset, num_workers, batch_size,
                            distributed, in_memory, num_small_crops):
        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()
        # First branch of augmentations
        self.decoder = ffcv.transforms.RandomResizedCrop((224, 224))
        image_pipeline_big: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))
        ]

        # Second branch of augmentations
        self.decoder2 = ffcv.transforms.RandomResizedCrop((224, 224))
        image_pipeline_big2: List[Operation] = [
            self.decoder2,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ffcv.transforms.RandomSolarization(0.2, 128),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        # SSL Augmentation pipeline
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]

        pipelines={
            'image': image_pipeline_big,
            'label': label_pipeline,
            'image_0': image_pipeline_big2
        }

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        custom_field_mapper={"image_0": "image"}

        # Add small crops (used for Dino)
        if num_small_crops > 0:
            self.decoder_small = ffcv.transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4))
            image_pipeline_small: List[Operation] = [
                self.decoder2,
                RandomHorizontalFlip(),
                ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
                ffcv.transforms.RandomGrayscale(0.2),
                ffcv.transforms.RandomSolarization(0.2, 128),
                ToTensor(),
                ToDevice(ch.device(this_device), non_blocking=True),
                ToTorchImage(),
                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
            ]
            for j in range(1,num_small_crops+1):
                pipelines["image_"+str(j)] = image_pipeline_small
                custom_field_mapper["image_"+str(j)] = "image"

        # Create data loader
        loader = ffcv.Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines=pipelines,
                        distributed=distributed,
                        custom_field_mapper=custom_field_mapper)


        return loader

    @param('data.num_workers')
    @param('training.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_train_loader_supervised(self,train_dataset, num_workers, batch_size,
                          resolution, distributed):
        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        order = OrderOption.SEQUENTIAL

        loader = ffcv.Loader(train_path,
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


    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, distributed):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
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

    @param('training.epochs')
    @param('logging.log_level')
    @param('training.eval_freq')
    def train(self, epochs, log_level, eval_freq):
        # We scale the number of max steps w.t the number of examples in the training set
        self.max_steps = epochs * self.num_train_exemples // (self.batch_size * self.world_size)
        for epoch in range(self.start_epoch, epochs):
            res = self.get_resolution(epoch)
            self.res = res
            self.decoder.output_size = (res, res)
            self.decoder2.output_size = (res, res)
            train_loss, stats =  self.train_loop_supervised(epoch)
            if log_level > 0:
                extra_dict = {
                    'train_loss': train_loss,
                    'epoch': epoch
                }
                self.log(dict(stats,  **extra_dict))
            if epoch % eval_freq == 0:
                # Eval and log
                self.eval_and_log(stats, extra_dict)
            # Empty cache
            ch.cuda.empty_cache()
            # Run checkpointing
            self.checkpoint(epoch+1)
        if self.gpu == 0:
            ch.save(self.model.state_dict(), self.log_folder / 'final_weights.pt')

    def eval_and_log(self, stats, extra_dict={}):
        stats = self.val_loop()
        self.log(dict(stats, **extra_dict))
        return stats

    @param('training.loss')
    def create_model_and_scaler(self, loss):
        scaler = GradScaler()
        model = SSLNetwork()
        '''
        if loss == "supervised":
#            model.fc = nn.Linear(model.num_features, self.num_classes)
            model.fc = nn.Sequential(
                        nn.BatchNorm1d(model.num_features), 
                        nn.ReLU(inplace = True), 
                        nn.Linear(model.num_features, self.num_classes)
                        )
        '''
        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
        return model, scaler

    @param('logging.checkpoint_freq')
    @param('logging.snapshot_freq')
    @param('training.train_probes_only')
    @param('data.random_seed')
    def checkpoint(self, epoch, checkpoint_freq, snapshot_freq, train_probes_only, random_seed):
        if self.rank != 0 or (epoch % checkpoint_freq != 0 and epoch % snapshot_freq != 0) :
            return
        if train_probes_only:
            state = dict(
                epoch=epoch, 
                probes=self.probes.state_dict(), 
                optimizer_probes=self.optimizer_probes.state_dict()
            )
            save_name = f"probes_ep{epoch}.pth"
            ch.save(state, self.log_folder / save_name)
        else:
            state = dict(
                epoch=epoch,
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict()
            )
            if epoch % snapshot_freq == 0: 
                save_name = f"model_ep{epoch}.pth" 
                ch.save(state, self.log_folder / save_name)
            if epoch % checkpoint_freq == 0: 
                save_name = f"model.pth" 
                ch.save(state, self.log_folder / save_name)

    @param('logging.log_level')
    @param('training.loss')
    @param('training.base_lr')
    @param('training.end_lr_ratio')
    @param('training.num_small_crops')
    def train_loop_supervised(self, epoch, log_level, loss, base_lr, end_lr_ratio, num_small_crops, aug=True):
        """
        Main training loop for SSL training with VicReg criterion.
        """
        model = self.model
        model.train()
        losses = []
        iterator = tqdm(self.train_loader)
        print('Entering training loop ...')
        print('Augmentation: ', aug)
        if aug:
            for ix, loaders in enumerate(iterator, start=epoch * len(self.train_loader)):
                # Get lr
                lr = learning_schedule(
                    global_step=ix,
                    batch_size=self.batch_size * self.world_size,
                    base_lr=base_lr,
                    end_lr_ratio=end_lr_ratio,
                    total_steps=self.max_steps,
                    warmup_steps=10 * self.num_train_exemples // (self.batch_size * self.world_size),
                )
                for g in self.optimizer.param_groups:
                    g["lr"] = lr

                # Get first view
                images_big_0 = loaders[0]
                labels_big = loaders[1]
                batch_size = loaders[1].size(0)
                images_big_1 = loaders[2]
                images_big = ch.cat((images_big_0, images_big_1), dim=0)
                self.optimizer.zero_grad(set_to_none=True)
                with autocast():
                    output = model(images_big_0.repeat(2,1,1,1))
                    loss_train = self.classif_loss(output, labels_big.repeat(2))
                    self.scaler.scale(loss_train).backward()
                    # TODO access the gradients for the layers here. Square them and sum
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # Logging
                if log_level > 0:
                    self.train_meters['loss'](loss_train.detach())
                    self.train_meters['acc'](output.softmax(dim=-1).detach(), labels_big.repeat(2))
                    losses.append(loss_train.detach())
                    group_lrs = []
                    for _, group in enumerate(self.optimizer.param_groups):
                        group_lrs.append(f'{group["lr"]:.5f}')

                    names = ['ep', 'iter', 'shape', 'lrs']
                    values = [epoch, ix, tuple(images_big.shape if aug else images.shape), group_lrs]
                    if log_level > 1:
                        names += ['loss']
                        values += [f'{loss_train.item():.3f}']

                    msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                    iterator.set_description(msg)

        else:
            for ix, (images, target) in enumerate(iterator, start=epoch * len(self.train_loader)):

                # Get lr
                lr = learning_schedule(
                    global_step=ix,
                    batch_size=self.batch_size * self.world_size,
                    base_lr=base_lr,
                    end_lr_ratio=end_lr_ratio,
                    total_steps=self.max_steps,
                    warmup_steps=10 * self.num_train_exemples // (self.batch_size * self.world_size),
                )
                for g in self.optimizer.param_groups:
                    g["lr"] = lr
                self.optimizer.zero_grad(set_to_none=True)
                with autocast():
                    output_classif_fc = model(images)
                    loss_train = self.classif_loss(output_classif_fc, target)
                
                    self.scaler.scale(loss_train).backward()
                    # TODO access the gradients for the layers here. Square them and sum
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

        # Return epoch's log
        if log_level > 0:
            self.train_meters['time'](ch.tensor(iterator.format_dict["elapsed"]))
            loss = ch.stack(losses).mean().cpu()
            assert not ch.isnan(loss), 'Loss is NaN!'
            stats = {k: m.compute().item() for k, m in self.train_meters.items()}
            [meter.reset() for meter in self.train_meters.values()]
            return loss.item(), stats


    def val_loop(self):
        model = self.model
        model.eval()
        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    output_classif_fc = model(images)
                    loss_val_fc = self.classif_loss(output_classif_fc, target)
                    self.val_meters['loss_val_fc'](loss_val_fc.detach())
                    pred_labels = ch.argmax(output_classif_fc.softmax(dim=-1).detach(), dim = -1)
                    print('target: ', target)
                    print('pred_labels: ', pred_labels)
                    inner_counter = {t.item(): 1 for t, cond in zip(target, target == pred_labels) if cond}
                    self.class_counter += inner_counter
                    print('output_classif_fc softmax: ', output_classif_fc.softmax(dim=-1).detach(), 'target: ', target)
                    self.val_meters['loss_val_fc_acc'](output_classif_fc.softmax(dim=-1).detach(), target)
                    self.loss_val_fc_mcss(output_classif_fc.softmax(dim=-1).detach(), target)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        stats['class_counter'] = copy.deepcopy(self.class_counter)
        # taking only true positives with
        tp = self.loss_val_fc_mcss.compute()[:,0]
        print('tp: ', tp)
        loss_val_fc_mcss_indices = ch.nonzero(tp, as_tuple=True)[0]

        stats['loss_val_fc_mcss_indices'] = loss_val_fc_mcss_indices.cpu().data.numpy().tolist()
        stats['loss_val_fc_mcss_values'] = tp[loss_val_fc_mcss_indices].cpu().data.numpy().tolist()

        [meter.reset() for meter in self.val_meters.values()]
        self.loss_val_fc_mcss.reset()
        self.class_counter.clear()

        return stats

    @param('logging.folder')
    def initialize_logger(self, folder):
        self.train_meters = {
            'loss': torchmetrics.MeanMetric().to(self.gpu),
            'acc': torchmetrics.Accuracy('multiclass',num_classes=1000,top_k=1).to(self.gpu),
            'time': torchmetrics.MeanMetric().to(self.gpu),
        }
        self.val_meters = {
            'loss_val_fc': torchmetrics.MeanMetric().to(self.gpu),
            'loss_val_fc_acc': torchmetrics.Accuracy('multiclass',num_classes=1000,top_k=1).to(self.gpu)
        }
        self.class_counter = Counter()
        self.loss_val_fc_mcss = MulticlassStatScores(num_classes=1000, average=None).to(self.gpu)

        if self.gpu == 0:
            if Path(folder + 'final_weights.pt').is_file():
                self.uid = ""
                folder = Path(folder)
            else:
                folder = Path(folder)
            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)
        self.log_folder = Path(folder)

    @param('training.train_probes_only')
    @param('data.random_seed')
    def log(self, content, train_probes_only, random_seed):
        print(f'=> Log: {content}')
        if self.rank != 0: return
        cur_time = time.time()
        name_file = 'log'
        with open(self.log_folder / name_file, 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    @param('dist.port')
    def launch_from_args(cls, distributed, world_size, port):
        if distributed:
            ngpus_per_node = ch.cuda.device_count()
            world_size = int(os.getenv("SLURM_NNODES", "1")) * ngpus_per_node
            if "SLURM_JOB_NODELIST" in os.environ:
                cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
                host_name = subprocess.check_output(cmd).decode().splitlines()[0]
                dist_url = f"tcp://{host_name}:"+port
            else:
                dist_url = "tcp://localhost:"+port
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=ngpus_per_node, join=True, args=(None, ngpus_per_node, world_size, dist_url))
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        if args[1] is not None:
            set_current_config(args[1])
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    @param('training.eval_only')
    @param('logging.folder')
    def exec(cls, gpu, config, ngpus_per_node, world_size, dist_url, distributed, eval_only, folder):
        trainer = cls(gpu=gpu, ngpus_per_node=ngpus_per_node, world_size=world_size, dist_url=dist_url)
        print('starting training loop: ', eval_only)
        if eval_only:
            trainer.eval_and_log()
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()

class Trainer(object):
    def __init__(self, config, num_gpus_per_node, dump_path, dist_url, port):
        self.num_gpus_per_node = num_gpus_per_node
        self.dump_path = dump_path
        self.dist_url = dist_url
        self.config = config
        self.port = port

    def __call__(self):
        self._setup_gpu_args()

    def checkpoint(self):
        self.dist_url = get_init_file().as_uri()
        empty_trainer = type(self)(self.config, self.num_gpus_per_node, self.dump_path, self.dist_url, self.port)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        from pathlib import Path
        job_env = submitit.JobEnvironment()
        self.dump_path = Path(str(self.dump_path).replace("%j", str(job_env.job_id)))
        gpu = job_env.local_rank
        world_size = job_env.num_tasks
        if "SLURM_JOB_NODELIST" in os.environ:
            cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
            host_name = subprocess.check_output(cmd).decode().splitlines()[0]
            dist_url = f"tcp://{host_name}:"+self.port
        else:
            dist_url = "tcp://localhost:"+self.port
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        ImageNetTrainer._exec_wrapper(gpu, config, self.num_gpus_per_node, world_size, dist_url)

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast SSL training')
    parser.add_argument("folder", type=str)
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()
    return config

@param('logging.folder')
@param('dist.ngpus')
@param('dist.nodes')
@param('dist.timeout')
@param('dist.partition')
@param('dist.comment')
@param('dist.port')
def run_submitit(config, folder, ngpus, nodes,  timeout, partition, comment, port):
    Path(folder).mkdir(parents=True, exist_ok=True)
    #create folder for NN attack data 
    (Path(folder) / 'attack').mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=folder, slurm_max_num_timeout=30)

    num_gpus_per_node = ngpus
    nodes = nodes
    timeout_min = timeout

    kwargs = {}
    #kwargs['slurm_comment'] = comment
    kwargs['slurm_constraint'] = 'volta32gb'
    
    executor.update_parameters(
        mem_gb=60 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node, 
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="ffcv2")

    dist_url = get_init_file().as_uri()

    trainer = Trainer(config, num_gpus_per_node, folder, dist_url, port)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {folder}")

@param('dist.use_submitit')
def main(config, use_submitit):
    if use_submitit:
        run_submitit(config)
    else:
        ImageNetTrainer.launch_from_args()

if __name__ == "__main__":
    config = make_config()
    main(config)
