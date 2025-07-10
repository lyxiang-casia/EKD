# [ICCV 2025] Evidential Knowledge Distillation




## Abstract

Existing logit-based knowledge distillation methods typically employ singularly deterministic categorical distributions, which eliminates the inherent uncertainty in network predictions and thereby limiting the effective transfer of knowledge. To address this limitation, we introduce distribution-based probabilistic modeling as a more comprehensive representation of network knowledge. Specifically, we regard the categorical distribution as a random variable and leverage deep neural networks to predict its distribution, representing it as an evidential second-order distribution. Based on the second-oder modeling, we propose Evidential Knowledge Distillation (EKD) which distills both the expectation of the teacher distribution and the distribution itself into the student. The expectation captures the macroscopic characteristics of the distribution, while the distribution itself conveys microscopic information about the classification boundaries. Additionally, we theoretically demonstrate that EKD's distillation objective provides an upper bound on the expected risk of the student when the teacher’s predictions are treated as ground truth labels. Extensive experiments on several standard benchmarks across various teacher-student network pairs highlight the effectiveness and superior performance of EKD.


## Usage (To be updated)

The code is built on [mdistiller](<https://github.com/megvii-research/mdistiller>), [Multi-Level-Logit-Distillation](<https://github.com/Jin-Ying/Multi-Level-Logit-Distillation>), [CTKD](<https://github.com/zhengli97/CTKD>) and [tiny-transformers](<https://github.com/lkhl/tiny-transformers>).


### Installation

Environments:

- Python 3.8
- PyTorch 1.7.0

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python setup.py develop
```

## Distilling CNNs

### CIFAR-100

- Download the [`cifar_teachers.tar`](<https://github.com/megvii-research/mdistiller/releases/tag/checkpoints>) and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.


1. For KD

  ```bash
  # KD
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml
  # KD+Ours
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```

2. For DKD

  ```bash
  # DKD
  python tools/train.py --cfg configs/cifar100/dkd/resnet32x4_resnet8x4.yaml 
  # DKD+Ours
  python tools/train.py --cfg configs/cifar100/dkd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```
3. For MLKD

  ```bash
  # MLKD
  python tools/train.py --cfg configs/cifar100/mlkd/resnet32x4_resnet8x4.yaml
  # MLKD+Ours
  python tools/train.py --cfg configs/cifar100/mlkd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```

4. For CTKD

Please refer to [CTKD](./CTKD).


### Training on ImageNet

- Download the dataset at <https://image-net.org/> and put it to `./data/imagenet`

  ```bash
  # KD
  python tools/train.py --cfg configs/imagenet/r34_r18/kd.yaml
  # KD+Ours
  python tools/train.py --cfg configs/imagenet/r34_r18/kd.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```

## Distilling ViTs

Please refer to [tiny-transformers](./tiny-transformers).

## Visualizations
Please refer to [visualizations](./tools/visualizations).

# Acknowledgement
Sincere gratitude to the contributors of mdistiller, CTKD, Multi-Level-Logit-Distillation and tiny-transformers for their distinguished efforts.


# :mega: Citation

If you find that this project helps your research, please consider citing some of the following paper:


