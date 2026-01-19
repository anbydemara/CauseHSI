# CauseHSI
This project provides code implementation for the paper "CauseHSI: Cross-scene Hyperspectral Image Classification via Causal Disentanglement".

# 1. Usage

## 1.1 Preparation

### 1.1.1 Requirements

```shell
pip intsall -r requirements.txt
```

Recommended environment:
- python 3.8
- pytorch 1.8

### 1.1.2 data structure

```
├── data
│   └── datasets
│       ├── Houston
│       ├── Pavia
│       └── HyRANK
└── ...
```

### 1.1.3 Pre-Augmented
```shell
python scg_aug.py
```

## 1.2 Train

- Houston:

```shell
python train-gan.py --data_path Houston/ --source_domain Houston13 --target_domain Houston18 \
  --training_sample_ratio 0.8 --flip_augmentation --radiation_augmentation
```

- Pavia:

```shell
python train-gan.py --data_path Pavia/ --source_domain PaviaU --target_domain PaviaC \
  --training_sample_ratio 0.5
```

- HyRANK:

```shell
python train-gan.py --data_path HyRANK/ --source_domain Dioni --target_domain Loukia \
  --training_sample_ratio 0.8
```

# 2. Acknowledgments
- The dataset is sourced from: https://github.com/YuxiangZhang-BIT/Data-CSHSI
