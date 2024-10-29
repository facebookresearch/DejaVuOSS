# Déjà Vu OSS for Out of the Box (OOB) Models

Déjà Vu OSS for OOB models allows us to detect and measure memorization in out of the box pretrained machine learning.
It is inspired by [DejaVu](https://github.com/facebookresearch/DejaVu) paper and allows to perform one model memorization tests without requiring to retrain an additional model for spurious correlation detection.

## Installation

**Installation Requirements**
- Python >= 3.6
- PyTorch >= 2.0.0

##### Installing 

**with latest pip and conda release

Descibe pip install here ... 

**with `conda`**

Create a new conda environment and activate it.

```
conda create -f environment.yml

conda activate dejavuoss

```

Below we describe how to train ResNet and Naive Bayes datast-level correlation detection classifiers.

## Training ResNet Correlation Classifier
### Generating background crops for images
To generate the background crops for ImageNet run:
```
python dejavuoss/crop_images.py --save_pth <DATA-HOME>/dejavu/imagenet/300_per_class_B \
                                --bbox_idx_pth <DATA-HOME>/imagenet_partition_out/300_per_class/bbox_B.npy \
                                --imgnet_pth <IMAGENET-HOME>/train/  \
                                --imgnet_annot_pth <DATA-HOME>/imagenet/annotation \
                                --imgnet_cls_pth <DATA-HOME>/imagenet/imgnet_classes.json
```

Train ResNet using [background_crop: forground_label] dataset. Make sure to convert the dataset into beton format before running the training.
See: https://github.com/facebookresearch/DejaVu for instructions.
```
bash_examples/1_training_vicreg_supervised_crops.sh
```

### Evaluating ResNet Correlation Classifier

```
python eval_crop.py --model_path <LOGGING_FOLDER>/resnet50_crops/supervised_train_A_test_B_wd_0.1_lars/model_ep75.pth \
                    --crop_dir <DATA-HOME>/dejavu/imagenet/300_per_class_B/crops  \
                    --resnet50 \
                    --save_path  <LOGGING_FOLDER>/resnet50_crops/eval                          

```

## Training Naive Bayes Classifier
We use Grounding Segment Anything to annotate the background crops. Install https://github.com/IDEA-Research/Grounded-Segment-Anything in a separate folder and create separate conda enviorment for it as described in the README. 

Run `grounding_dino_imagenet_annotator.py` in the Grounded-Segment-Anything's conda enviroment for both training and test datasets using `is_train` flag.
```
python nb_counts_generator.py --crop_dir <DATA-HOME>/dejavu/imagenet/300_per_class_B/crops \
                              --imagenet_label_path <IMAGENET-HOME>/labels.txt \
                              --is_train 1 \
                              --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
                              --ram_checkpoint ram_swin_large_14m.pth \
                              --grounded_checkpoint groundingdino_swint_ogc.pth \
                              --sam_checkpoint sam_vit_h_4b8939.pth \
                              --output_dir "output_trainA_testB_frac_0.5" \
                              --box_threshold 0.25 \
                              --text_threshold 0.2 \
                              --iou_threshold 0.5 \
                              --device "cuda" \
                              --frac 0.5

```
The above command will generate the features and frequency counts for background crops.

Run Naive Bayes correlation detector:

```
python memorization_nb.py --input_path <OUTPUT-HOME>/outputs_nb_trainA_testB \
					      --output_path <OUTPUT-HOME>/outputs_nb_trainA_testB \
  						  --imagenet_label_path <IMAGENET-HOME>/labels.txt \
						  --label_topk 3  \
 						  --smooth 1e-10 
```

### Evaluating Naive Bayes (NB) Correlation Classifier
`memorization_nb.py` above also prints the evaluation results - accuracy of the NB classifier


## Running the Attack on VicReg OSS model

The attack is conducted similar to the original [DejaVu](https://github.com/facebookresearch/DejaVu) paper.

python label_inference_attack_vicreg_oob.py --local 0 \
                                            --resnet50 \
                                            --loss vicreg \
                                            --output_dir <LOGGING_FOLDER>/vicreg/attack_sweeps/NN_attk_vicreg \
                                            --public_idx_pth $INDEX_FOLDER/300_per_class/public.npy  \
                                            --test_idx_pth $INDEX_FOLDER/300_per_class/bbox_A.npy  \
                                            --valid_idx_pth $INDEX_FOLDER/300_per_class/bbox_B.npy \
                                            --imgnet_train_pth $IMAGENET_DATASET_TRAIN_DIR  \
                                            --imgnet_valid_pth $IMAGENET_DATASET_TRAIN_DIR   \
                                            --imgnet_bbox_pth  $IMAGENET_BBOX_ANNOTATIONS \
                                            --imgnet_valid_bbox_pth  $IMAGENET_BBOX_ANNOTATIONS \
                                            --use_backbone 0  \
                                            --k 100 \
                                            --k_attk 100 

## A list of ImageNet ids with highest dataset-level correlations based on the ResNet50 and NB correlation detection clasifiers.
The list can be found under:
```
data\dataset_level_correlations_indices.npy
```
The list is currated based on ResNet50 and Naive Bayes correlation detection classifiers trained on 300k examples (300 per class) chosen uniformly random from ImageNet training dataset.
We share a list of dataset-level correlations based the intesection of two ResNet50-based correlation detection models trained with different random initialization seeds and a Naive Bayes classifer. Naive Bayes uses top 5 background annotations as input features.


## License

DejaVuOSS is licensed under the CC-BY-NC 4.0 license.