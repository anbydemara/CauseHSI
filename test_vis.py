#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CauseHSI 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2025/6/20 15:33 
"""
import argparse
import os

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.datasets import get_dataset, HyperX

from models.discriminator import CDFENet

from utils.data_util import sample_gt, seed_worker, metrics
from utils.train_util import get_metrics
from config.config import get_args

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def vis_res(outs, indice, name):
    # 输入参数

    h = args.height  # 原图高度（根据实际情况修改）
    w = args.width  # 原图宽度（根据实际情况修改）

    # 创建空矩阵并填充预测结果
    classification_map = np.zeros((h, w), dtype=outs.dtype)

    # 处理坐标越界并转换为整数
    rows = np.clip(indice[:, 0], 0, h - 1).astype(int)
    cols = np.clip(indice[:, 1], 0, w - 1).astype(int)

    # 将预测结果填入对应位置
    classification_map[rows, cols] = outs

    # 创建离散颜色映射
    # colors = plt.cm.get_cmap('tab10', num_class)(np.linspace(0, 1, num_class))
    # cmap = ListedColormap(colors)

    if args.target_domain == 'Loukia':
        custom_colors = ['#000000', '#f67088', '#e78230', '#b9962e', '#98a231', '#4faf31', '#008080', '#33cccc',
                         '#0066cc', '#6666ff', '#a68df5', '#e766f6', '#f667c1']
        # num_class = 13
    else:
        custom_colors = ['#000000', '#f67088', '#C59332', '#81a831', '#33af8a', '#36aabb', '#7f96f4',
                         '#f45dec']  # 十六进制或RGB字符串
        # num_class = 12
    cmap = ListedColormap(custom_colors)

    # 可视化设置
    dpi = 100  # 你可以根据需要调整 dpi
    figsize = (w / dpi, h / dpi)  # 计算 figsize

    plt.figure(figsize=figsize)
    img = plt.imshow(classification_map, cmap=cmap, interpolation='none')
    # plt.colorbar(img, ticks=[], shrink=0.8)
    # plt.colorbar(img, ticks=range(num_class), shrink=0.8)
    plt.axis('off')

    # 保存图像（支持png/jpg/pdf等格式）
    plt.savefig(f'{args.seed}-{name}.png',
                bbox_inches='tight',
                pad_inches=0,
                dpi=300,
                facecolor='auto')
    plt.close()


def reduce_dimensions(data, n_pca=50):
    """标准化 + PCA + t-SNE 降维"""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=n_pca)
    data_pca = pca.fit_transform(data_scaled)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    data_tsne = tsne.fit_transform(data_pca)
    return data_tsne


def plot_and_save(data_tsne, labels, title, filename,
                  cmap='tab10', dpi=200, figsize=(10, 8)):
    if args.target_domain == 'Loukia':
        custom_colors = ['#000000', '#f67088', '#e78230', '#b9962e', '#98a231', '#4faf31', '#008080', '#33cccc',
                         '#0066cc', '#6666ff', '#a68df5', '#e766f6', '#f667c1']
        # num_class = 13
    else:
        custom_colors = ['#000000', '#f67088', '#C59332', '#81a831', '#33af8a', '#36aabb', '#7f96f4', '#f45dec']

    cmap = ListedColormap(custom_colors)

    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        data_tsne[:, 0], data_tsne[:, 1],
        c=labels, cmap=cmap, alpha=0.7,
        edgecolors='w', linewidths=0.3,
        s=1  # 点大小
    )
    cbar = plt.colorbar(scatter, ticks=np.unique(labels))
    cbar.set_label('Class', rotation=270, labelpad=15)
    plt.title(title)
    plt.axis('off')  # 可选：若不需要坐标轴刻度可关闭
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')  # bbox_inches防止标签被截断
    plt.close()  # 显式关闭图形，避免内存泄漏


def evaluate(test_loader, indice):
    num_samples = 1000  # 目标采样数量

    model = CDFENet(in_channels=N_BANDS, embed_dim=args.embed, num_class=num_classes)

    if torch.cuda.is_available():
        state_dict = torch.load(f"./run/{args.source_domain}/ckpt/best.pkl")
        model.to(args.gpu)
    else:
        state_dict = torch.load(f"./run/{args.source_domain}/ckpt/best.pkl", map_location=torch.device('cpu'))

    model.load_state_dict(state_dict['Discriminator'])
    model.eval()
    loop = tqdm(test_loader, desc='Testing')
    with torch.no_grad():
        outs = []
        labels = []


        for i, (data, label) in enumerate(loop):
            label = label - 1
            data = data.to(args.gpu)
            out, feat = model(data)

            out = out.argmax(dim=1)
            outs.append(out.detach().cpu().numpy())
            labels.append(label.numpy())

        outs = np.concatenate(outs)
        labels = np.concatenate(labels)
        acc = np.mean(outs == labels) * 100
        vis_res(np.asarray(outs+1).astype(int), indice, f'{acc:.2f}')
        # vis_res(np.asarray(labels+1).astype(int), indice, 'groud_truth')
        results = metrics(outs, labels, n_classes=labels.max().astype(int) + 1)

        print('metrics [oa: {:2.2f} | kappa: {:2.2f} | aa: {:2.2f}] | TPR: {}'.format(
            results['Accuracy'],
            results['Kappa'] * 100,
            results['AA'] * 100, np.round(results['TPR'] * 100, 2)))


if __name__ == '__main__':
    # 全局参数 & 设置
    DATA_ROOT = './data/datasets/'
    args = get_args()
    hyperparams = vars(args)
    seed_worker(args.seed)

    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_domain,
                                                                                        os.path.join(DATA_ROOT,
                                                                                                     args.data_path))
    sample_num_tar = len(np.nonzero(gt_tar)[0])
    num_classes = gt_tar.max().astype(int)
    N_BANDS = img_tar.shape[-1]

    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full',
                        'height': gt_tar.shape[0], 'width': gt_tar.shape[1]})

    r = int(args.patch_size / 2) + 1
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')

    g = torch.Generator()
    g.manual_seed(args.seed)

    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    indice = test_dataset.indices - r
    test_loader = DataLoader(test_dataset,
                             pin_memory=True,
                             worker_init_fn=seed_worker,
                             generator=g,
                             batch_size=hyperparams['batch_size'])
    evaluate(test_loader, indice)