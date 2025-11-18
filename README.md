# [ICCV 2025] Evidential Knowledge Distillation




## Abstract

Existing logit-based knowledge distillation methods typically employ singularly deterministic categorical distributions, which eliminates the inherent uncertainty in network predictions and thereby limiting the effective transfer of knowledge. To address this limitation, we introduce distribution-based probabilistic modeling as a more comprehensive representation of network knowledge. Specifically, we regard the categorical distribution as a random variable and leverage deep neural networks to predict its distribution, representing it as an evidential second-order distribution. Based on the second-oder modeling, we propose Evidential Knowledge Distillation (EKD) which distills both the expectation of the teacher distribution and the distribution itself into the student. The expectation captures the macroscopic characteristics of the distribution, while the distribution itself conveys microscopic information about the classification boundaries. Additionally, we theoretically demonstrate that EKD's distillation objective provides an upper bound on the expected risk of the student when the teacher’s predictions are treated as ground truth labels. Extensive experiments on several standard benchmarks across various teacher-student network pairs highlight the effectiveness and superior performance of EKD.


## Usage

The code is built on [mdistiller](<https://github.com/megvii-research/mdistiller>), and [logit-standardization-KD](<https://github.com/sunshangquan/logit-standardization-KD>).

### Installation

Environments:

- Python 3.8
- PyTorch 1.7.0

Install the package:

```
pip install -r requirements.txt
python setup.py develop
```

## Training Evidential Teachers
- To use EKD, you should first train an evidential teacher to provide guidance. Use the following command to train an evidential teacher.: 

  ``` bash
  python tools/train.py --cfg configs/cifar100/teacher/resnet32x4.yaml
  ```

**Note**: A single training run does not guarantee that you will obtain an ideal pretrained teacher, as its performance may differ from that of a Softmax-based teacher. You may need to train multiple times or directly use the pretrained teacher we provide：

- Download the [`evidential_teachers.tar`](<https://drive.google.com/file/d/19fXKTl_2DsZpiR4SuzXOePG_slyLTMks/view?usp=sharing>) and untar it to `./evidential_teachers` via `tar xvf evidential_teachers.tar`.

## Distilling CNNs

### CIFAR-100 Dataset

#### Distillation
- Different teacher–student distillation settings can be achieved by modifying the configuration file `*.yaml` in the following command:
  ```bash
  # EKD
  python tools/train.py --cfg configs/cifar100/ekd/resnet32x4_resnet8x4.yaml 
  ```
  If the dataset does not exist locally, it will be fetched during the first run.



### ImageNet Dataset

#### Download Dataset
- Download the dataset at <https://image-net.org/> and put it to `./data/imagenet`

#### Distillation
Use the following command to train on the **ImageNet** dataset:
  ```bash
  # ResNet34-ResNet18
  python tools/train.py --cfg configs/imagenet/r34_r18/ekd.yaml
  # ResNet50-MobileNetV2
  python tools/train.py --cfg configs/imagenet/r50_mv2/ekd.yaml
  ```


# Acknowledgement
Sincere gratitude to the contributors of [mdistiller](https://github.com/megvii-research/mdistiller) and [logit-standardization-KD](https://github.com/megvii-research/mdistiller) for their distinguished efforts.


# Citation

If you find that this project helps your research, please consider citing some of the following paper:

```bibtex
@inproceedings{xiang2025evidential,
  title={Evidential Knowledge Distillation},
  author={Xiang, Liangyu and Gao, Junyu and Xu, Changsheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2814--2824},
  year={2025}
}


