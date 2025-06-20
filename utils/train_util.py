#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CauseHSI 
@File    ：train_util.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2025/6/20 15:33 
"""
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

def save_model(model, best_acc, epoch, folder, best=False):
    """
    保存模型
    :param models:
    :param optimizers:
    :param val_acc:
    :param epoch:
    :param best:
    :return:
    """
    save_path = f'run/{folder}/ckpt/'
    model_params = {
        'epoch': epoch,
        'best_acc': best_acc,
        # 'extractor_state_dict': models['extractor'].state_dict(),
        # 'classifier_state_dict': models['classifier'].state_dict(),
        'model_state_dict': model.state_dict(),
        # 'generator_state_dict': models['generator'].state_dict()
    }
    if best:
        save_path = os.path.join(save_path, 'best_{}.pth.tar'.format(best_acc))
    else:
        save_path = os.path.join(save_path, 'last.pth.tar')
        # 当没有更改优化器参数时（动态调整学习率，weight_decay等等），优化器状态不变，可以不保存
        # model_params.update({
        #     # 'extractor_optimizer_dict': optimizers['extractor'].state_dict(),
        #     # 'classifier_optimizer_dict': optimizers['classifier'].state_dict(),
        #     # 'model_optimizer_dict': optimizers['model'].state_dict(),
        #     # 'generator_optimizer_dict': optimizers['generator'].state_dict()
        # })
    torch.save(model_params, save_path)


def adjust_learning_rate(optimizer, epoch, origin_lr):
    """
    动态调整学习率：没50轮缩小10倍
    """
    lr = origin_lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def correct_num(out, labels):
    assert out.shape[0] == labels.shape[0]  # 保证批次数量一致
    _, pred = out.max(dim=1)    # 预测值
    return torch.sum(pred.eq(labels))  # 正确数量


def get_metrics(pred, label):
    metrics = {}
    # 1. confusion matrix
    matrix = confusion_matrix(label, pred, labels=np.arange(7))
    metrics['matrix'] = matrix
    # 2. OA
    oa = compute_oa(matrix)
    metrics['oa'] = oa
    # 3. Kappa
    kappa = compute_kappa(matrix)
    metrics['kappa'] = kappa
    # 4. AA
    aa = compute_aa(matrix)
    metrics['aa'] = aa
    # 5. Recall
    # 6. Precesion
    # 7. F1
    # 8. ..
    # print(metrics['matrix'])
    return metrics

def compute_oa(matrix):
    """
    计算总体准确率OA: OA = (TP+TN)/(TP+TN+FP+FN)
    :param matrix: 混淆矩阵
    :return: OA
    """
    return np.trace(matrix) / np.sum(matrix)
def compute_aa(matrix):
    """
    计算平均准确率AA
    :param matrix: 混淆矩阵
    :return: AA
    """
    return np.mean(np.diag(matrix) / np.sum(matrix, axis=1))

def compute_kappa(matrix):
    """
    计算Kappa系数
    :param matrix: 混淆矩阵
    :return: Kappa系数
    """
    oa = compute_oa(matrix)
    pe = 0
    for i in range(len(matrix)):
        pe += np.sum(matrix[i]) * np.sum(matrix[:, i])
    pe /= np.sum(matrix) ** 2
    return (oa - pe) / (1 - pe)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def factorization_loss(f_a, f_b):
    # empirical cross-correlation matrix
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0)+1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0)+1e-6)
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    loss = on_diag + 0.005 * off_diag

    return loss

def ACE(featureidx, numsamples, classifier, feature, device):
    bs, zdim = feature.shape
    zdo = torch.randn(numsamples, bs, zdim).to(device)
    zdo[:, :, featureidx] = feature[:, featureidx]
    sample = classifier(zdo.view(numsamples * bs, zdim))
    ACEdo = sample.view(numsamples, bs, -1).mean(0)

    zrand = torch.randn(numsamples, bs, zdim).to(device)
    sample = classifier(zrand.view(numsamples * bs, zdim))
    ACEbaseline = sample.view(numsamples, bs, -1).mean(0)
    ace = ACEbaseline - ACEdo
    return ace

def contrastive_ace(numsamples, classifier, feature, anchorbs, device):
    numfeature = feature.shape[1]
    ace = []
    for i in range(numfeature):
        ace.append(ACE(i, numsamples, classifier, feature, device))

    acematrix = torch.stack(ace, dim=1) / (
                torch.stack(ace, dim=1).norm(dim=1).unsqueeze(1) + 1e-8)  # [bs, num_feature]
    anchor = acematrix[:anchorbs] / acematrix[:anchorbs].norm(1)
    neighbor = acematrix[anchorbs:2 * anchorbs] / acematrix[anchorbs:2 * anchorbs].norm(1)
    distant = acematrix[2 * anchorbs:] / acematrix[2 * anchorbs:].norm(1)

    margin = 0.02
    pos = (torch.abs(anchor - neighbor)).sum()
    neg = (torch.abs(anchor - distant)).sum()
    contrastive_loss = F.relu(pos - neg + margin)

    return contrastive_loss