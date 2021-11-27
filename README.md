# MIL-VT

Code for MICCAI 2021 accepted paper: MIL-VT: Multiple Instance Learning Enhanced Vision Transformer for Fundus Image Classification

### Basic Requirement:
* timm==0.3.2
* torch==1.7.0
* torchvision==0.8.1
* vit-pytorch==0.6.6
* numpy==1.19.5
* opencv-python==4.5.1.48
* pandas==1.1.5
* imgaug==0.4.0



### Pretrain Weight for MIL-VT on large fundus dataset
* Please download pretrained weight of fundus image from this link:
* https://drive.google.com/drive/folders/1YgdhA7BK6Unrs2lOflOd9rPTrwm17gdf?usp=sharing
* Store the pretrain weight in 'weights/fundus_pretrained_VT_small_patch16_384_5Class.pth.tar'


### Dataset
* APTOS data from kaggle: https://www.kaggle.com/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy
* RFMiD data from IEEE data port: https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid

### BibTex:
```
@inproceedings{yu2021mil,
  title={Mil-vt: Multiple instance learning enhanced vision transformer for fundus image classification},
  author={Yu, Shuang and Ma, Kai and Bi, Qi and Bian, Cheng and Ning, Munan and He, Nanjun and Li, Yuexiang and Liu, Hanruo and Zheng, Yefeng},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={45--54},
  year={2021},
  organization={Springer}
}
```
