# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os, gc

import numpy as np
import csv
import random
import json
import torch
import torchvision
from PIL import Image

from wordcloud import WordCloud, STOPWORDS
from collections import Counter

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
) 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
import ram
from ram.models.ram import RAM #ram
from ram.models.utils import load_checkpoint_swinbase, load_checkpoint_swinlarge, load_checkpoint
#from ram import inference_ram
import torchvision.transforms as TS
from pathlib import Path

stopwords = set(STOPWORDS)
CONFIG_PATH=(Path(ram.models.utils.__file__).resolve().parents[1])
print('CONFIG_PATH: ', CONFIG_PATH)

class RAM_Wrapper(RAM):
    def __init__(self,
                 med_config=f'{CONFIG_PATH}/configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 prompt='a picture of ',
                 threshold=0.68,
                 delete_tag_index=[],
                 tag_list=f'{CONFIG_PATH}/data/ram_tag_list.txt',
                 tag_list_chinese=f'{CONFIG_PATH}/data/ram_tag_list_chinese.txt',
                 stage='eval'):
        super().__init__(med_config=med_config,
                 image_size=image_size,
                 vit=vit,
                 vit_grad_ckpt=vit_grad_ckpt,
                 vit_ckpt_layer=vit_ckpt_layer,
                 prompt=prompt,
                 threshold=threshold,
                 delete_tag_index=delete_tag_index,
                 tag_list=tag_list,
                 tag_list_chinese=tag_list_chinese,
                 stage=stage)
    
    def generate_tag(self,
                    image,
                    threshold=0.68,
                    tag_input=None,
                    ):
            label_embed = torch.nn.functional.relu(self.wordvec_proj(self.label_embed))

            image_embeds = self.image_proj(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1],
                                    dtype=torch.long).to(image.device)

            # recognized image tags using image-tag recogntiion decoder
            image_cls_embeds = image_embeds[:, 0, :]
            image_spatial_embeds = image_embeds[:, 1:, :]

            bs = image_spatial_embeds.shape[0]
            label_embed = label_embed.unsqueeze(0).repeat(bs, 1, 1)
            tagging_embed = self.tagging_head(
                encoder_embeds=label_embed,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=False,
                mode='tagging',
            )

            logits = self.fc(tagging_embed[0]).squeeze(-1)

            targets = torch.where(
                torch.sigmoid(logits) > self.class_threshold.to(image.device),
                torch.tensor(1.0).to(image.device),
                torch.zeros(self.num_class).to(image.device))

            targets_scores = torch.sigmoid(logits)
            targets_scores = torch.where(
                targets_scores > self.class_threshold.to(image.device),
                targets_scores,
                torch.zeros(self.num_class).to(image.device))

            tag = targets.cpu().numpy()
            tag_scores = targets_scores.cpu().numpy()
            #tag[:, self.delete_tag_index] = 0
            #tag_scores[:, self.delete_tag_index] = 0
            tag_output = []
            tag_scores_lst = []
            for b in range(bs):
                index = np.argwhere(tag[b] == 1)
                if len(index) == 0:
                    continue
                token = self.tag_list[index].squeeze(axis=1)
                score = tag_scores[0][index]
 
                tag_output.append(token)
                tag_scores_lst.append(score)

            return tag_output, tag_scores_lst

# load RAM pretrained model parameters
def ram_local(pretrained='', **kwargs):
    model = RAM_Wrapper(**kwargs)
    if pretrained:
        if kwargs['vit'] == 'swin_b':
            model, msg = load_checkpoint_swinbase(model, pretrained, kwargs)
        elif kwargs['vit'] == 'swin_l':
            model, msg = load_checkpoint_swinlarge(model, pretrained, kwargs)
        else:
            model, msg = load_checkpoint(model, pretrained)
#         print('msg', msg)
    return model

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def predict_crop_labels(ram_model, transform, input_path, image_path, device):
        image_path_full = os.path.join(input_path, image_path)
        # load image
        image_pil, _ = load_image(image_path_full)

        raw_image = image_pil.resize(
                        (384, 384))
        raw_image  = transform(raw_image).unsqueeze(0).to(device)

        #res = inference_ram(raw_image , ram_model)
        with torch.no_grad():
            tags, scores = ram_model.generate_tag(raw_image)
        if len(tags) > 0:
            return tags[0].tolist(), scores[0]
        else:
            return [], []


def compute_and_store_predictions(ram_model, transform, input_path, output_dir, cls_id, cls_name, device):
    input_path = os.path.join(input_path, str(cls_id))

    if not os.path.exists(input_path):
        print('No examples available for class: ', cls_id, 'class name: ', cls_name)
        return

    file_names = os.listdir(input_path)

    labels = []
    for file_name in file_names:
        labels_dict = {}
        tags, scores = predict_crop_labels(ram_model, transform, input_path, file_name, device)

        labels_dict['image_path'] = file_name
        labels_dict['tags'] = [tag for tag in tags]
        labels_dict['scores'] = [float(score[0]) for score in scores]
        labels.append(labels_dict)

    # python save predicted labels
    output_dir_labels =  os.path.join(output_dir, 'labels')
    if not os.path.exists(output_dir_labels):
        os.mkdir(output_dir_labels)
    
    labels_path = os.path.join(output_dir_labels, f'{cls_id}_{cls_name}_labels.json')
    with open(labels_path, 'w') as fp:
        json.dump(labels, fp)


def compute_and_store_frequencies(ram_model, transform, input_path, output_dir,
                                  cls_id, cls_name, device, frac=1.0):
    input_path = os.path.join(input_path, str(cls_id))
    if not os.path.exists(input_path):
        print('No examples available for class: ', cls_id, 'class name: ', cls_name)
        return

    # python save frequences
    output_dir_freq =  os.path.join(output_dir, str(frac), 'frequencies')
    if not os.path.exists(output_dir_freq):
        os.mkdir(output_dir_freq)
    
    freq_path = os.path.join(output_dir_freq, f'{cls_id}_{cls_name}_freq.json')
    if os.path.exists(freq_path):
        print('Frequencies for class: ', cls_id, 'class name: ', cls_name, ' are already generated. Path: ', freq_path)

    # HERE we iterate over the dataset crops
    image_paths = os.listdir(input_path)
    num_files_image_paths = len([name for name in image_paths])
    selected_frac = round(num_files_image_paths * frac)
    tags_all = ''
    tag2score_map = Counter()
    tag2freq_map = Counter()
    for image_path in random.sample(image_paths, selected_frac): 
        tags, scores = predict_crop_labels(ram_model, transform, input_path, image_path, device)
        if len(tags) == 0:
            continue
        tags_all = " ".join([tags_all, ','.join(tags)])
        for tag, score in zip(tags, scores):
            tag2score_map[tag] += score[0]
            tag2freq_map[tag] += 1
    
    wordcloud_obj = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10)
    frequences = wordcloud_obj.process_text(tags_all)
    wordcloud = wordcloud_obj.generate_from_frequencies(frequences)

    with open(freq_path, 'w') as fp:
        json.dump({tag2freq[0]: [tag2freq[1], tag2score[1]] for tag2freq, tag2score in \
            zip(tag2freq_map.items(), tag2score_map.items())}, fp)

    # python wordcloud of labels
    output_dir_wordcloud =  os.path.join(output_dir, 'wordclouds', str(frac))
    if not os.path.exists(output_dir_wordcloud):
        os.mkdir(output_dir_wordcloud)
    
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig(
            os.path.join(output_dir_wordcloud, f'{cls_name}_wordcloud.JPEG'), 
            bbox_inches="tight", dpi=300, pad_inches=0.0
    )

def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--crop_dir", required=True, type=str, help="path to crop directory")
    parser.add_argument("--imagenet_label_path", required=True, type=str, help="path to imagenet label mapping")
    parser.add_argument("--is_train", type=int, required=True, help="a flag whether the pipeline is run for training dataset")

    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--ram_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    # parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--openai_key", type=str, help="key for chatgpt")
    parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--frac", type=float, default="1.0", help="The random fraction of "
                            "examples used for training")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    ram_checkpoint = args.ram_checkpoint  # change the path of the model
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    # image_path = args.input_image
    split = args.split
    openai_key = args.openai_key
    openai_proxy = args.openai_proxy
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device
    is_train = args.is_train
    crop_dir = args.crop_dir
    imagenet_label_path = args.imagenet_label_path
    frac = args.frac

    print('is_train: ', args.is_train)
    
    torch.cuda.empty_cache()
    gc.collect()

    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                        TS.Resize((384, 384)),
                        TS.ToTensor(), normalize
                    ])
        
    # load model
    ram_model = ram_local(pretrained=ram_checkpoint,
                          image_size=384,
                          vit='swin_l')
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    ram_model.eval()

    ram_model = ram_model.to(device)
    print(ram_model)

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # imagenet folder labels

    with open(imagenet_label_path) as folder2label:
        folder2label_reader = csv.reader(folder2label, delimiter=',')
        folder2label_lst = [line for line in folder2label_reader]

    print('output_dir: ', output_dir)
    if is_train == 1:
        print('train')
        for folder2label in folder2label_lst:
            # compute and store frequences and work cloud from train dataset
            compute_and_store_frequencies(ram_model, transform, crop_dir, output_dir, folder2label[0], folder2label[1],  device, frac)
    else:
        print('inference')
        # compute and store predicted labels from valid dataset
        for folder2label in folder2label_lst:
            compute_and_store_predictions(ram_model, transform, crop_dir, output_dir, folder2label[0], folder2label[1], device)
        