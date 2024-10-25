# Déjà Vu OSS for Out of the Box (OOB) Models

Déjà Vu OSS for OOB models allows us to detect and measure memorization in out of the box pretrained machine learning.
It is inspired by DejaVu(https://github.com/facebookresearch/DejaVu) paper and allows to perform one model memorization tests without requiring to retrain an additional model for spurious correlation detection.

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
conda create -n dejavuoss -f environment.yml

conda activate dejavuoss

```

## A list of ImageNet ids with highest dataset-level correlations based on the ResNet50 and Naive Bayes correlation detection clasifiers.
The list can be found under:
```
data\dataset_level_correlations_indices.npy
```
The list is currated based on ResNet50 and Naive Bayes correlation detection classifiers trained on 300k examples (300 per class) chosen uniformly random from ImageNet training dataset.
We share a list of dataset-level correlations based the intesection of two ResNet50-based correlation detection models trained with different random initialization seeds and a Naive Bayes classifer. Naive Bayes uses top 5 background annotations as input features.


## License

DejaVuOSS is licensed under the CC-BY-NC 4.0 license.