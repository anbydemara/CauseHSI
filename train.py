#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CauseHSI 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2025/6/20 15:30 
"""

import os
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from config.config import get_args
from data.datasets import get_dataset, HyperX, DoubleHyper


from models.discriminator import CDFENet


from models.generator import Generator
from utils.data_util import sample_gt, seed_worker, metrics



"""
先优化discriminator，再优化generator
分类损失、独立性损失 + 重构损失 + loss_consist

生成域的非因果+源域的因果 --> 重构生成域 
"""
def train(epoch):
    model.train()
    train_loss = 0  # loss
    for it, (data, label, data_aug, label_aug) in enumerate(train_loader):
        data, label = data.to(args.gpu), label.to(args.gpu).long()
        data_aug = data_aug.to(args.gpu)

        label -= 1

        # 优化discriminator
        with torch.no_grad():
            x_ED = generator(data, data_aug)

        # domain_labels = torch.cat((torch.zeros(data.shape[0]), torch.ones(x_ED.shape[0])), dim=0).to(args.gpu).long()

        x = torch.cat((data, x_ED), dim=0)

        labelD = label.repeat(2)

        # loss_class, loss_indep, loss_complete, loss_consist_feat, loss_consist_rec = model(x, labelD, domain_labels, mode='train')

        loss_class, loss_indep, loss_complete, loss_consist_feat, loss_consist_rec = model(x, labelD, mode='train')
        loss = loss_class + args.lambda_1 * (loss_indep + loss_complete + loss_consist_feat + loss_consist_rec)

        M_opt.zero_grad()
        loss.backward()
        M_opt.step()

        # 优化generator

        # 冻结 model 的参数
        model.requires_grad_(False)

        x_TD = generator(data, data_aug)
        x = torch.cat((data, x_TD), dim=0)
        # domain_labels = torch.cat((torch.zeros(data.shape[0]), torch.ones(x_TD.shape[0])), dim=0).to(args.gpu).long()

        # balance_loss, loss_class_TD, loss_domains_TD = model(x, labelD, domain_labels, mode='train', domain='TD')
        # loss = balance_loss + loss_class_TD + loss_domains_TD

        # loss_control, loss_class_TD = model(x, labelD, domain_labels, mode='train', domain='TD')
        loss_control, loss_class_TD = model(x, labelD, mode='train', domain='TD')
        loss = args.lambda_2 * loss_control + loss_class_TD

        G_opt.zero_grad()
        loss.backward()
        G_opt.step()

        model.requires_grad_(True)

        # 打印统计信息
        train_loss += loss.item()
        writer.add_scalar('loss_class', loss_class, epoch)

        writer.add_scalar('loss_indep', loss_indep, epoch)
        writer.add_scalar('loss_complete', loss_complete, epoch)
        writer.add_scalar('loss_consist_feat', loss_consist_feat, epoch)
        writer.add_scalar('loss_control', loss_control, epoch)
        writer.add_scalar('loss_class_TD', loss_class_TD, epoch)
        writer.add_scalar('loss_consist_rec', loss_consist_rec, epoch)


def validation(best_oa=0):
    model.eval()
    ps = []
    ys = []
    for i, (x1, y1) in enumerate(test_loader):
        y1 = y1 - 1
        with torch.no_grad():
            x1 = x1.to(args.gpu)
            p1, _ = model(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys == ps) * 100
    writer.add_scalar('oa', acc, epoch)
    results = metrics(ps, ys, n_classes=ys.max().astype(int) + 1)
    print('TPR: {} | current OA: {:2.2f} | best OA: {:2.2f} | AA: {:2.2f} | Kappa: {:2.2f}'.format(
        np.round(results['TPR'] * 100, 2), results['Accuracy'], best_oa, results['AA'] * 100, results['Kappa'] * 100))
    return results


if __name__ == '__main__':
    # 全局参数 & 设置
    DATA_ROOT = './data/datasets/'
    args = get_args()
    hyperparams = vars(args)
    seed_worker(args.seed)

    ## log
    root = os.path.join(args.save_path, args.source_domain)
    # sub_dir = 'seed_' + str(args.seed) + '_lr' + str(args.lr) + '_ps' + str(args.patch_size) + '_bs' + str(
    #    args.batch_size) + '_' + datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
    sub_dir = 'seed_' + str(args.seed) + '_embed_' + str(args.embed) + '_' + datetime.strftime(datetime.now(),
                                                                                               '%m-%d_%H-%M-%S')
    log_dir = os.path.join(str(root), sub_dir)

    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    # 数据加载
    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_domain,
                                                                                        os.path.join(DATA_ROOT,
                                                                                                     args.data_path))
    img_scg, gt_scg, LABEL_VALUES_scg, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_domain + '_A',
                                                                                        os.path.join(DATA_ROOT,
                                                                                                     args.data_path))
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_domain,
                                                                                        os.path.join(DATA_ROOT,
                                                                                                     args.data_path))
    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_scg = len(np.nonzero(gt_scg)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])

    tmp = args.training_sample_ratio * args.re_ratio * sample_num_src / sample_num_tar

    num_classes = gt_src.max().astype(int)
    N_BANDS = img_src.shape[-1]
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})

    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_scg = np.pad(img_scg, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_scg = np.pad(gt_scg, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    train_gt_src, val_gt_src, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    train_gt_scg, _, _, _ = sample_gt(gt_scg, args.training_sample_ratio, mode='random')
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')

    img_src_con, train_gt_src_con = img_src, train_gt_src
    img_scg_con, train_gt_scg_con = img_scg, train_gt_scg

    val_gt_src_con = val_gt_src

    if tmp < 1:
        for i in range(args.re_ratio - 1):
            img_src_con = np.concatenate((img_src_con, img_src))
            img_scg_con = np.concatenate((img_scg_con, img_scg))
            train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
            train_gt_scg_con = np.concatenate((train_gt_scg_con, train_gt_scg))
            val_gt_src_con = np.concatenate((val_gt_src_con, val_gt_src))

    hyperparams_train = hyperparams.copy()
    g = torch.Generator()
    g.manual_seed(args.seed)
    # train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    # scg_dataset = HyperX(img_scg, train_gt_scg, **hyperparams_train)
    double_dataset = DoubleHyper(img_src_con, train_gt_src_con, img_scg_con, train_gt_scg_con, **hyperparams_train)
    train_loader = DataLoader(double_dataset,
                              batch_size=hyperparams['batch_size'],
                              pin_memory=True,
                              worker_init_fn=seed_worker,
                              generator=g,
                              shuffle=True)  # , drop_last=True
    # val_dataset = HyperX(img_src_con, val_gt_src_con, **hyperparams)
    # val_loader = DataLoader(val_dataset,
    #                             pin_memory=True,
    #                             batch_size=hyperparams['batch_size'])
    hyperparams.update({'flip_augmentation': False, 'radiation_augmentation': False})
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = DataLoader(test_dataset,
                             pin_memory=True,
                             worker_init_fn=seed_worker,
                             generator=g,
                             batch_size=hyperparams['batch_size'])
    imsize = [hyperparams['patch_size'], hyperparams['patch_size']]

    # 生成器&优化器
    generator = Generator(n=args.d_se, imdim=N_BANDS, imsize=imsize, zdim=10)
    G_opt = Adam(generator.parameters(), lr=args.lr, weight_decay=args.l2_lambda)


    # new model
    model = CDFENet(in_channels=N_BANDS, embed_dim=args.embed, num_class=num_classes)
    M_opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_lambda)

    cls_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()

    # device_ids = [0, 1]
    if torch.cuda.is_available():
        # model = nn.DataParallel(model, device_ids=device_ids)
        # generator = nn.DataParallel(generator, device_ids=device_ids)
        model.to(args.gpu)
        generator.to(args.gpu)

    best_oa = 0
    for epoch in range(1, args.epoch + 1):
        start = time.time()
        train(epoch)
        end = time.time()
        print(f'Epoch:{epoch}/{args.epoch} time:{end - start} bn:{len(train_loader)}')
        start = time.time()
        results = validation(best_oa=best_oa)
        end = time.time()
        print(f'test time: {end - start}')
        if best_oa < results['Accuracy']:
            best_oa = results['Accuracy']
            torch.save({'Discriminator': model.state_dict()}, os.path.join(log_dir, f'best.pkl'))
            with open(os.path.join(log_dir, 'best_oa.txt'), 'w', encoding='utf-8') as file:
                file.write(
                    'TPR: {} | OA: {:2.2f} | AA: {:2.2f} | Kappa: {:2.2f} | lambda1={}, lambda2={}, embed={}'.format(
                        np.round(results['TPR'] * 100, 2), results['Accuracy'], results['AA'] * 100,
                                                                                results['Kappa'] * 100, args.lambda_1,
                        args.lambda_2, args.embed))
        print()
    writer.close()