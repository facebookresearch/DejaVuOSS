# Move the content of visualize_valid_save_crops.ipynb

import time
import torch, torchvision, json, os
import numpy as np
import torchvision
import argparse
from utils.image_common import AuxDataset, InverseTransform
from pathlib import Path
from PIL import Image
from collections import Counter

r'''

Command to run the script:

python dejavuoob/crop_images.py --save_pth OUTPU_DIR \
    --bbox_idx_pth PATH_TO_A_FILE_WITH_IMAGENET_INDEX_SUBSETS \
    --imgnet_pth PATH_TO_YOUR_IMAGENET_FOLDER \
    --imgnet_annot_pth PATH_TO_YOUR_IMAGENET_ANNOT_FOLDER \
    --imgnet_cls_pth PATH_TO_YOUR_IMAGENET_CLS_JSON

'''

class SSL_Transform: 
    """Transform applied to SSL examples at test time 
    """
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.ssl_xfrm = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                        ])
        
    def __call__(self, x): 
       return self.ssl_xfrm(x)

class SSL_Transform2:
    def __init__(self):
        resolution = 224
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

def save_images(iTrans, patch, sample, root_path, folder, fname):
    patch = iTrans(patch)
    patch = patch.squeeze().permute(1,2,0).numpy()
    save_folder =  os.path.join(root_path, 'crops', folder)
    #if not os.path.exists(save_folder):
    #    os.mkdir(save_folder)
    # don't save final image since it can be time consuming
    final_img = Image.fromarray((patch * 255).astype(np.uint8))
    final_img.save(os.path.join(save_folder, fname), "JPEG")

    sample = sample.squeeze().permute(1,2,0).numpy()
    save_folder =  os.path.join(root_path, 'original', folder)
    #if not os.path.exists(save_folder):
    #    os.mkdir(save_folder)
    final_img = Image.fromarray((sample * 255).astype(np.uint8))
    final_img.save(os.path.join(save_folder, fname), "JPEG")

def main(args):
    save_pth = args.save_pth
    imgnet_cls_pth = args.imgnet_cls_pth
    bbox_idx_pth = args.bbox_idx_pth
    imgnet_pth = args.imgnet_pth
    imgnet_annot_pth = args.imgnet_annot_pth

    # create save paths if they do not exist
    if not os.path.exists(save_pth / 'original'):
        os.makedirs(save_pth / 'original')
    if not os.path.exists(save_pth / 'crops'):
        os.makedirs(save_pth / 'crops')

    # load imagenet classes
    with open(imgnet_cls_pth) as f:
        imgnet_classes = json.load(f)

    # initialize logger
    log_fn = log_outer(save_pth, 'log')

    # read image indices
    bbox_idx = np.load(bbox_idx_pth)

    crop_ds = AuxDataset(imgnet_pth, imgnet_annot_pth, bbox_idx, return_im_and_tgt = True, log_fn = log_fn)

    iTrans = InverseTransform()

    tgt_counts = Counter()
    for i, (patch, good, sample, tgt) in enumerate(crop_ds): 
        if good == -1:
            continue
        with torch.no_grad():
            index = crop_ds.indices[i]
            path, target = crop_ds.samples[index]
            print('path: ', path, 'target: ', target)
            print('sample: ', sample.shape)

            cls = path.split('/')[-2]
            fname = path.split('/')[-1].split('.')[0] + '.JPEG'

            print('fname: ', fname)
            print('cls: ', cls, 'label:', imgnet_classes[tgt])

            # save path and original 
            if not os.path.exists(save_pth / 'original' / cls):
                os.mkdir(save_pth / 'original' / cls)

            if not os.path.exists(save_pth / 'crops' / cls):
                os.mkdir(save_pth / 'crops' / cls)

            save_images(iTrans, patch, sample, save_pth, cls, fname)
            tgt_counts[tgt] += 1

    # save tgt level counts
    summary_path = save_pth / 'summary.json'
    with open(summary_path, 'w') as fp:
        json.dump(tgt_counts, fp)

def log_outer(log_folder, name_file):
    logging_path = log_folder / name_file

    def log(content):
        cur_time = time.time()
        with open(logging_path, 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                **content
            }) + '\n')
            fd.flush()
    return log

def parse_args():
    parser = argparse.ArgumentParser("Arguments for extracting and saving crops")

    parser.add_argument("--save_pth", type=Path, help = 'root path to store crops and '
                                                        'a copy of original images')
    parser.add_argument("--bbox_idx_pth", type=Path)
    parser.add_argument("--imgnet_pth", type=Path, help="root directory to imagenet") 
    parser.add_argument("--imgnet_annot_pth", type=Path, help="root directory to imagenet foreground annotations") 
    parser.add_argument("--imgnet_cls_pth", type=Path, help="imagenet class path") 

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
