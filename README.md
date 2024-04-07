# Provable Defense against Adversarial Attacks to Multi-modal Models

## Introduction
This is the official PyTorch implementation of [**MMCert: Provable Defense against Adversarial Attacks to Multi-modal Models**](https://arxiv.org/abs/2403.19080), accepted by [CVPR 2024].
In this paper, we propose an independent subsampling strategy that allows us to provably defend against adversarial perturbations in all modalities. Our method is model agnostic. Below is an illustration of this subsampling strategy, where we subsample ablated versions of different modalities independently.

<p align="center">
<img src="/subsampling (1).png" width="80%"/>
</p>

In this repo, we implement certified segmentation for the [KITTI Road Dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) using [SNE-RoadSeg](https://github.com/hlwang1124/SNE-RoadSeg) as the multi-modal model. We test our code in Python 3.8, CUDA 12.3, and PyTorch 2.2.2.
## Setup
Please setup the KITTI Road Dataset and models trained with ablated inputs according to the following folder structure:
```
MultimodalCertification
 |-- checkpoints
 |  |-- kitti
 |  |  |-- CertificationMethod_net_RoadSeg.pth
 |-- data
 |-- datasets
 |  |-- kitti
 |  |  |-- calib
 |  |  |-- depth_u16
 |  |  |-- gt_image_2
 |  |  |-- image_2
 |-- output
 |-- figs
 ...
```
Please check [SNE-RoadSeg](https://github.com/hlwang1124/SNE-RoadSeg) for more details about the KITTI Road Dataset. `image_2`, `gt_image_2` and `calib` can be downloaded from the [KITTI Road Dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php). `depth_u16` is based on the LiDAR data provided in the KITTI Road Dataset, and it can be downloaded from [here](https://drive.google.com/file/d/16ft3_V8bMM-av5khZpP6_-Vtz-kF6X7c/view?usp=sharing). Since the original testing set does not contain ground-truth labels, we split the original training set into a new training set and a testing set. The `output` folder is used to store the output of test_ensemble.py, which contains all the base model predictions for these ablated multi-modal inputs. They will later be used for certification purpose in certify.ipynb.

## Usage

### Training on the KITTI dataset
For training with ablated inputs, you first need to setup the `datasets/kitti` folder as mentioned above, and then run:
```
python train.py --dataroot datasets/kitti --dataset kitti --use_sne --certification_method MMCert --ablation_ratio_train 0.05
```
and the model weights will be saved in `checkpoints`. Note that `use-sne` in `train.sh` means the SNE model is used.

### Testing on the KITTI dataset
The next step is to create ablated versions of testing inputs, and use the trained base model to make predictions for them. For the default setting, you run:
```
python test_ensemble.py --dataroot datasets/kitti --dataset kitti --use_sne --certification_method MMCert
```
, and you will get all base model predictions for these ablated versions of testing inputs in `output`. If you want to compare with the baseline method (randomized ablation), just replace "MMCert" with "randomized_ablation" for both training and testing.

### Certification on the KITTI dataset
Then, we can analyze the certification performance of our method using the information saved from the last step. You can use `certify.ipynb` to try different certification settings.

## Citation
You can cite our paper if you use this code for your research.
```
@article{wang2024mmcert,
  title={MMCert: Provable Defense against Adversarial Attacks to Multi-modal Models},
  author={Wang, Yanting and Fu, Hongye and Zou, Wei and Jia, Jinyuan},
  journal={arXiv preprint arXiv:2403.19080},
  year={2024}
}
```

## Acknowledgement
Our code is based on [SNE-RoadSeg](https://github.com/hlwang1124/SNE-RoadSeg).
