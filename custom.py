#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CauseHSI 
@File    ：custom.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2025/6/20 15:24 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedLoss(nn.Module):
    def __init__(self, alpha=1, beta=1e4, gamma=1e-3):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def content_loss(self, features1, features2):
        return self.mse(features1, features2)

    def style_loss(self, style_feat1, style_feat2):
        def gram_matrix(x):
            # batch, ch, h, w = x.size()
            # x = x.view(batch, ch, -1)
            gram = torch.mm(x.t(), x)  # [D, B] @ [B, D] → [D, D]
            gram = gram / x.size(0)  # 批次平均

            # return torch.bmm(x, x.transpose(1, 2)) / (ch * h * w)
            return gram

        gram1 = gram_matrix(style_feat1)
        gram2 = gram_matrix(style_feat2)
        return self.mse(gram1, gram2)

    def variation_loss(self, img):
        return torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
            torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

    def forward(self, imgs, content_feats, style_feats):
        img1, img2 = imgs
        content_feat1, content_feat2 = content_feats
        style_feat1, style_feat2 = style_feats

        c_loss = self.content_loss(content_feat1, content_feat2)
        s_loss = self.style_loss(style_feat1, style_feat1)
        v_loss = self.variation_loss(img2)
        total_loss = self.alpha * c_loss + self.beta * s_loss + self.gamma * v_loss
        return total_loss


class ControlledEmbeddingLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1,
                 min_thresh=0.2, max_thresh=0.5):
        super().__init__()
        self.alpha = alpha  # 内容损失权重
        self.beta = beta  # 风格损失权重
        self.gamma = gamma  # 差异约束权重
        self.min_thresh = min_thresh  # 最小差异阈值
        self.max_thresh = max_thresh  # 最大差异阈值

    def gram_matrix(self, x):
        # x形状: [B, D]
        return torch.mm(x.t(), x) / x.size(0)

    def content_loss(self, orig, gen):
        # return F.mse_loss(orig, gen)
        return 1 - F.cosine_similarity(orig, gen).mean()

    def style_loss(self, orig, gen):
        gram_orig = self.gram_matrix(orig)
        gram_gen = self.gram_matrix(gen)
        return F.mse_loss(gram_orig, gram_gen)

    def constraint_loss(self, orig, gen):
        diff = torch.norm(orig - gen, dim=1)  # L2距离 [B]
        loss_min = torch.relu(self.min_thresh - diff).mean()  # 差异过小惩罚
        loss_max = torch.relu(diff - self.max_thresh).mean()  # 差异过大惩罚
        return loss_min + loss_max

    # def forward(self, content_orig, content_gen, style_orig, style_gen):
    def forward(self, style_orig, style_gen):
        # 计算各损失项
        # L_content = self.content_loss(content_orig, content_gen)
        L_style = self.style_loss(style_orig, style_gen)
        # L_constraint = self.constraint_loss(content_orig, content_gen)  # 可选：约束内容或风格
        L_constraint = self.constraint_loss(style_orig, style_gen)  # 可选：约束内容或风格

        # 总损失
        # total_loss = (
        #     self.alpha * L_content +
        #     self.beta * L_style +
        #     self.gamma * L_constraint
        # )
        total_loss = (
                self.beta * L_style +
                self.gamma * L_constraint
        )
        # return {
        #     "total_loss": total_loss,
        #     "content_loss": L_content,
        #     "style_loss": L_style,
        #     "constraint_loss": L_constraint
        # }
        return total_loss