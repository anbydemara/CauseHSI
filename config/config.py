#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CauseHSI 
@File    ：config.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2025/6/20 15:20 
"""
import argparse


def get_args(parser=argparse.ArgumentParser(description='YoungNet parser')):
    # 数据集参数
    parser.add_argument('--source_domain', type=str, default='PaviaU', help='the name of source domain and data file')
    parser.add_argument('--target_domain', type=str, default='PaviaC', help='the name of target domain and data file')

    # 存储路径参数
    parser.add_argument('--data_path', type=str, default='Pavia/', help='the folder name of datasets')
    parser.add_argument('--save_path', type=str, default='./run', help='the path of results')

    # 训练参数
    # parser.add_argument('--epoch', type=int, default=500, help='The number of epoch')
    parser.add_argument('--epoch', type=int, default=400, help='The number of epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--l2_lambda', type=float, default=1e-4, help='L2-norm regularization')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--patch_size', type=int, default=13, help='Size of the input')
    parser.add_argument('--seed', type=int, default=233, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='default device gpu:0')
    # parser.add_argument('--pro_dim', type=int, default=128)
    parser.add_argument('--pro_dim', type=int, default=512)
    parser.add_argument('--training_sample_ratio', type=float, default=0.8)  # Houston

    # parser.add_argument('--training_sample_ratio', type=float, default=0.5) # Pavia
    parser.add_argument('--re_ratio', type=int, default=5)
    parser.add_argument('--d_se', type=int, default=64)
    # parser.add_argument('--d_se', type=int, default=256)
    parser.add_argument('--lambda_1', type=float, default=1.0)
    parser.add_argument('--lambda_2', type=float, default=1.0)
    parser.add_argument('--lr_scheduler', type=str, default='none')

    # 断点训练参数
    parser.add_argument('--resume', type=bool, default=False, help='Resume the training')

    # Houston数据集：Flip and radiation is True
    # parser.add_argument('--flip_augmentation', action='store_true', default=True,
    #                     help="Random flips (if patch_size > 1)")
    # parser.add_argument('--radiation_augmentation', action='store_true', default=True,
    #                     help="Random radiation noise (illumination)")
    # parser.add_argument('--mixture_augmentation', action='store_true', default=False,
    #                     help="Random mixes between spectra")

    parser.add_argument('--flip_augmentation', action='store_true',
                        help="Random flips (if patch_size > 1)")  # default=False
    parser.add_argument('--radiation_augmentation', action='store_true',
                        help="Random radiation noise (illumination)")  # default=False
    parser.add_argument('--mixture_augmentation', action='store_true',
                        help="Random mixes between spectra")  # default=False

    return parser.parse_args()