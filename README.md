# Provable Defense against Adversarial Attacks to Multi-modal Models

## Introduction
This is the official PyTorch implementation of [**MMCert: Provable Defense against Adversarial Attacks to Multi-modal Models**](https://arxiv.org/abs/2403.19080), accepted by [CVPR 2024].
In this paper, we propose an independent subsampling strategy that allows us to provably defend against adversarial perturbations in all modalities. Our method is model agnostic. Below is an illustration of this subsampling strategy, where we subsample ablated versions of different modalities independently.

<p align="center">
<img src="/subsampling (1).png" width="80%"/>
</p>

In this repo, we provide a certified segmentation example for the [KITTI Road Dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) using [SNE-RoadSeg](https://github.com/hlwang1124/SNE-RoadSeg) as the multi-modal model. We test our code in Python 3.8, CUDA 12.3, and PyTorch 2.2.2.
## Setup
Please setup the KITTI Road Dataset and models trained with ablated inputs according to the following folder structure:
```
MultimodalCertification
 |-- checkpoints
 |  |-- kitti
 |  |  |-- MethodName_net_RoadSeg.pth
 |-- data
 |-- datasets
 |  |-- kitti
 |  |  |-- training
 |  |  |  |-- calib
 |  |  |  |-- depth_u16
 |  |  |  |-- gt_image_2
 |  |  |  |-- image_2
 |-- output
 ...
```
Please check [SNE-RoadSeg](https://github.com/hlwang1124/SNE-RoadSeg) for more details about the KITTI Road Dataset. `image_2`, `gt_image_2` and `calib` can be downloaded from the [KITTI Road Dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php). `depth_u16` is based on the LiDAR data provided in the KITTI Road Dataset, and it can be downloaded from [here](https://drive.google.com/file/d/16ft3_V8bMM-av5khZpP6_-Vtz-kF6X7c/view?usp=sharing). Since the original testing set does not contain ground-truth labels, we split the original training set into a new training set and a testing set. The output folder is used to store the output of test_ensemble.py, which contains all ablated multi-modal inputs and their corresponding model predictions. They will later be used for certification purpose in certify.ipynb.

## Usage

### Training on the KITTI dataset
For training, you need to setup the `datasets/kitti` folder as mentioned above.
```
bash ./scripts/train.sh
```
and the weights will be saved in `checkpoints` and the tensorboard record containing the loss curves as well as the performance on the validation set will be save in `runs`. Note that `use-sne` in `train.sh` controls if we will use our SNE model, and the default is True. If you delete it, our RoadSeg will take depth images as input, and you also need to delete `use-sne` in `test.sh` to avoid errors when testing.

### Testing on the KITTI dataset
For KITTI submission, you need to setup the `checkpoints` and the `datasets/kitti/testing` folder as mentioned above. Then, run the following script:
```
bash ./scripts/test.sh
```
and you will get the prediction results in `testresults`. After that you can follow the [submission instructions](http://www.cvlibs.net/datasets/kitti/eval_road.php) to transform the prediction results into the BEV perspective for submission.

If everything works fine, you will get a MaxF score of **96.74** for **URBAN**. Note that this is our re-implemented weights, and it is very similar to the reported ones in the paper (a MaxF score of **96.75** for **URBAN**).

### Certification on the KITTI dataset
For KITTI submission, you need to setup the `checkpoints` and the `datasets/kitti/testing` folder as mentioned above. Then, run the following script:
```
bash ./scripts/test.sh
```
and you will get the prediction results in `testresults`. After that you can follow the [submission instructions](http://www.cvlibs.net/datasets/kitti/eval_road.php) to transform the prediction results into the BEV perspective for submission.

If everything works fine, you will get a MaxF score of **96.74** for **URBAN**. Note that this is our re-implemented weights, and it is very similar to the reported ones in the paper (a MaxF score of **96.75** for **URBAN**).

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
