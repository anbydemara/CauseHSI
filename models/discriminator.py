#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CauseHSI 
@File    ：discriminator.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2025/6/20 15:22 
"""
import torch
from torch import nn
import torch.nn.functional as F
from models.hsic import HSIC
from torch_dct import dct, idct, dct_2d, idct_2d

from custom import ControlledEmbeddingLoss


class SSMF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSMF, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.branch2 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.after_layer = nn.Sequential(
            nn.Conv2d(3 + 64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, x):
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)

        feat = torch.cat((feat1, feat2), dim=1)
        feat = self.after_layer(feat)
        return feat


class SCSS_Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SCSS_Conv, self).__init__()
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.depth_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_channels,
            # bias=False
        )

        self.leaky = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        out = self.point_conv(self.BN(x))
        out = self.leaky(out)
        out = self.depth_conv(out)
        out = self.relu(out)

        return out


class Extractor(nn.Module):
    def __init__(self, embed_dim=512, num_class=7):
        super(Extractor, self).__init__()

        self.sematic_encoder = nn.Sequential(
            SCSS_Conv(in_channels=128, out_channels=128, kernel_size=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            SCSS_Conv(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
        )

        self.domain_encoder = nn.Sequential(
            SCSS_Conv(in_channels=128, out_channels=128, kernel_size=1),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            SCSS_Conv(in_channels=128, out_channels=128, kernel_size=3),

            # nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2, groups=128),  # depthwise
            nn.Conv2d(128, 128, kernel_size=1),  # pointwise
            nn.ReLU()
        )

        self.afterlayer1 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.BatchNorm1d(128),
            nn.Linear(128, embed_dim),
        )

        self.afterlayer2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.BatchNorm1d(128),
            nn.Linear(128, embed_dim),
        )
        # self.classifier = Classifier(embed_dim, num_class)
        self.classifier = nn.Linear(embed_dim, num_class)

        # self.domain_classifier = Classifier(embed_dim, 2)
        self.domain_classifier = nn.Linear(embed_dim, 2)

    def forward(self, x, mode='test'):
        feat_class = self.sematic_encoder(x)
        feat_class = self.afterlayer1(feat_class)

        if mode == 'test':
            return self.classifier(feat_class), feat_class
            # return self.classifier(feat_class)
        else:
            feat_domain = self.domain_encoder(x)
            feat_domain = self.afterlayer2(feat_domain)

            pre_class = self.classifier(feat_class)
            # pre_domains = self.domain_classifier(feat_domain)

            # return pre_class, pre_domains, feat_class, feat_domain
            return pre_class, feat_class, feat_domain


class Decoder(nn.Module):
    def __init__(self, in_dim=1024, out_channels=102):
        super(Decoder, self).__init__()
        # input (bs, 1024)
        self.fcLayer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        self.convTransLayer = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1),  # (bs, 128, 13, 13)
            nn.ReLU(),
        )
        self.convLayer = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fcLayer(x).unsqueeze(2).unsqueeze(3)
        x = self.convTransLayer(x)
        x = self.convLayer(x)
        return x


class CDFENet(nn.Module):
    def __init__(self, in_channels=102, embed_dim=512, num_class=7):
        super(CDFENet, self).__init__()
        self.ssmf = SSMF(in_channels=in_channels, out_channels=in_channels)
        self.extractor = Extractor(embed_dim=embed_dim, num_class=num_class)

        # self.proj_head = nn.Linear(512, 128)

        self.decoder = Decoder(in_dim=embed_dim * 2, out_channels=in_channels)

        self.cls_criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
        self.controledLoss = ControlledEmbeddingLoss()

    def combineIn(self, caus, nonc):
        dim = caus.shape[1] // 2
        spec_caus = dct(caus, norm='ortho')
        spec_nonc = dct(nonc, norm='ortho')

        low = spec_nonc[:, :dim]
        mid = spec_caus  # 中频用 spec_a 全部内容
        high = spec_nonc[:, dim:]

        spec_combined = torch.cat([low, mid, high], dim=-1)  # 1024
        feat_combined = idct(spec_combined, norm='ortho')
        return feat_combined

    def forward(self, x, label=None, mode='test', domain='SD'):
        original_image = x
        x = self.ssmf(x)

        if mode == 'train':

            # pre_class, pre_domains, z_class, z_domains = self.extractor(x, mode)
            pre_class, z_class, z_domains = self.extractor(x, mode)
            bach_size = z_class.shape[0] // 2

            sd_causal, td_causal = z_class[:bach_size], z_class[bach_size:]
            sd_nonc, td_nonc = z_domains[:bach_size], z_domains[bach_size:]
            causal_recon = self.combineIn(sd_causal, td_nonc)
            caus_x = self.decoder(causal_recon)
            loss_consist_rec = self.mae_criterion(caus_x, original_image[bach_size:])

            if domain == 'TD':
                loss_control = self.controledLoss(sd_nonc, td_nonc)

                # loss_domains = self.cls_criterion(pre_domains[bach_size:], domain_label[bach_size:])
                loss_class = self.cls_criterion(pre_class[bach_size:], label[bach_size:])

                # return loss_class, recon_loss, loss_domains
                return loss_control, loss_class

            recon_feat = self.combineIn(z_class, z_domains)
            recon_x = self.decoder(recon_feat)

            loss_complete = self.mae_criterion(recon_x, original_image)

            z_class_normalized = F.normalize(z_class, p=2, dim=1)
            z_domains_normalized = F.normalize(z_domains, p=2, dim=1)

            loss_class = self.cls_criterion(pre_class, label)

            loss_indep = HSIC(z_class_normalized, z_domains_normalized)  # gpu 1
            # oth_loss1 = HSIC(z_class_normalized, z_domains_normalized.detach()) # gpu 0

            batch = z_class.shape[0] // 2
            loss_consist_feat = self.mse_criterion(z_class[:batch], z_class[batch:])

            return loss_class, loss_indep, loss_complete, loss_consist_feat, loss_consist_rec
            # return loss
        else:
            return self.extractor(x, mode)