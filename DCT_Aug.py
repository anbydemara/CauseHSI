#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CauseHSI 
@File    ：DCT_Aug.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2025/6/20 15:24 
"""
import numpy as np
from scipy.fftpack import dct, idct
import scipy.io as sio

import h5py


def dct2d(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')


def idct2d(dct_image):
    return idct(idct(dct_image.T, norm='ortho').T, norm='ortho')


def load_hyperspectral_image(mat_file_path):
    mat = sio.loadmat(mat_file_path)
    img = mat['ori_data']
    return img


def FreCom_hyper(img):
    h, w, c = img.shape
    img_dct = np.zeros((h, w, c))
    for i in range(c):
        img_ = img[:, :, i]  # 获取每个波段
        img_ = np.float32(img_)  # 将数值精度调整为32位浮点型
        img_dct[:, :, i] = dct2d(img_)  # 使用dct获得img的频域图像

    return img_dct


def Matching_hyper(img):
    # theta = np.random.uniform(alpha, beta)
    h, w, c = img.shape  # 954 210 48
    img_dct = FreCom_hyper(img)

    mask = np.zeros((h, w, c))
    v1 = int(min(h, w) * 0.005)  # 低中频划分
    v2 = int(min(h, w) * 0.7)  # 中高频划分
    v3 = min(h, w)

    # if v1 <= 0:
    #     raise ValueError("image too small")

    # 简便带通滤波器设计
    for x in range(h):
        for y in range(w):
            if (max(x, y) <= v1):
                mask[x][y] = 1 - max(x, y) / v1 * 0.95
            elif (v1 < max(x, y) <= v2):
                mask[x][y] = 0.01
            elif (v2 <= max(x, y) <= v3):
                mask[x][y] = (max(x, y) - v2) / (v3 - v2) * 0.3
            else:
                mask[x][y] = 0.5
    n_mask = 1 - mask
    # 划分为因果部分和非因果部分
    non_img_dct = img_dct * mask
    cal_img_dct = img_dct * n_mask

    # 非因果部分随即变换
    ref_dct = np.zeros_like(non_img_dct)
    for i in range(c):
        ref_dct[:, :, i] = non_img_dct[:, :, i] * (1 + np.random.randn())

    # 重新组合
    img_fc = ref_dct + cal_img_dct

    img_out = np.zeros((h, w, c))
    for i in range(c):
        img_out[:, :, i] = idct2d(img_fc[:, :, i]).clip(min_val, max_val)

    return img_out


def is_integer_2d_array(arr):
    """
    判断给定的二维数组是否是整数二维数组。

    参数:
    arr (list of lists): 要检查的二维数组。

    返回:
    bool: 如果二维数组中的所有元素都是整数，则返回 True；否则返回 False。
    """
    for sublist in arr:
        if not isinstance(sublist, list):
            return False  # 如果子元素不是列表，则不是二维数组
        for element in sublist:
            if not isinstance(element, int):
                print(element)
                return False
    return True


if __name__ == '__main__':
    np.random.seed(233)
    # mat_path = 'data/datasets/Pavia/PaviaU.mat'
    mat_path = 'data/datasets/Houston/Houston13.mat'
    # mat_path = './data/datasets/HyRANK/Dioni.mat'

    #### Pavia and Dioni start
    #     mat = sio.loadmat(mat_path)

    #     img = np.asarray(mat['ori_data'])  # .transpose(1, 2, 0)

    #     min_val = np.min(img)  # 高光谱图像的最小值
    #     max_val = np.max(img)  # 高光谱图像的最大值

    #     w, h, c = img.shape
    #     n = h // w
    #     img_blocks = []
    #     for i in range(n):
    #         if i == n - 1:
    #             img_i = img[:, i * w : , :]
    #         else:
    #             img_i = img[:, i * w: (i + 1) * w, :]
    #         img_blocks.append(img_i)

    #     # 对高光谱图像进行频域匹配处理
    #     # img_matched = Matching_hyper(img)

    #     img_matchs = []
    #     for i in range(n):
    #         img_matched_i = Matching_hyper(img_blocks[i])
    #         img_matchs.append(img_matched_i)
    #     img_matched = np.concatenate(img_matchs, axis=1)
    #     mat['ori_data'] = img_matched.astype(np.uint16)

    #     sio.savemat('./data/datasets/HyRANK/Dioni_A.mat', mat)
    #### Pavia and Dioni end

    #### Houston start
    with h5py.File(mat_path, 'r') as f_in:
        img = np.asarray(f_in['ori_data'])
        print(img.shape)
        img = img.transpose(2, 1, 0)
        print(img.shape)
        min_val = np.min(img)  # 高光谱图像的最小值
        max_val = np.max(img)  # 高光谱图像的最大值

        # 对高光谱图像进行频域匹配处理
        w, h, c = img.shape

        n = h // w
        img_blocks = []
        for i in range(n):
            if i == n - 1:
                img_i = img[:, i * w:, :]
            else:
                img_i = img[:, i * w: (i + 1) * w, :]

            img_blocks.append(img_i)

        # 对高光谱图像进行频域匹配处理
        # img_matched = Matching_hyper(img)

        img_matchs = []
        for i in range(n):
            img_matched_i = Matching_hyper(img_blocks[i])
            img_matchs.append(img_matched_i)
        img_matched = np.concatenate(img_matchs, axis=1)

        img_matched = img_matched.transpose(2, 1, 0)

        with h5py.File('./data/datasets/Houston/Houston13_A.mat', 'w') as f_out:
            f_out.create_dataset('ori_data', data=img_matched)
#### Houston end
# mat['ori_data'] = img_matched
# sio.savemat('./data/Pavia/PaviaU_A.mat', mat)
# sio.savemat('./data/datasets/Houston/Houston13_A.mat', mat)
# h5py.savemat('./data/datasets/Houston/Houston13_A.mat', mat)