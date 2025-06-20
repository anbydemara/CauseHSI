#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CauseHSI 
@File    ：generator.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2025/6/20 15:22 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpaRandomization(nn.Module):
    def __init__(self, num_features, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        # self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)

        self.raw_alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    def forward(self, x, ):
        N, C, H, W = x.size()
        # x = self.norm(x)
        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()

            idx_swap = torch.randperm(N)
            alpha = torch.rand(N, 1, 1)
            mean = self.alpha * mean + (1 - self.alpha) * mean[idx_swap]
            var = self.alpha * var + (1 - self.alpha) * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x, idx_swap


# class SpaRandomization(nn.Module):
#     # def __init__(self, num_features, eps=1e-5, device=0):
#     def __init__(self, num_features, eps=1e-5):
#         super().__init__()
#         self.eps = eps
#         self.norm = nn.InstanceNorm2d(num_features, affine=False)
#         # self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)
#         self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

#     def forward(self, x, ):
#         N, C, H, W = x.size()
#         # x = self.norm(x)

#         if self.training:
#             x = x.view(N, C, -1)
#             mean = x.mean(-1, keepdim=True)
#             var = x.var(-1, keepdim=True)
#             x = (x - mean) / (var + self.eps).sqrt()
#             idx_swap = torch.randperm(N)
#             # alpha = torch.rand(N, 1, 1)
#             mean = self.alpha * mean + (1 - self.alpha) * mean[idx_swap]
#             var = self.alpha * var + (1 - self.alpha) * var[idx_swap]
#             x = x * (var + self.eps).sqrt() + mean  # x 开始出现nan，原因时公式里的var < 0
#             x = x.view(N, C, H, W)

#         return x, idx_swap


class SpeRandomization(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, idx_swap, y=None):
        N, C, H, W = x.size()

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(1, keepdim=True)
            var = x.var(1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()
            if y != None:
                for i in range(len(y.unique())):
                    index = y == y.unique()[i]
                    tmp, mean_tmp, var_tmp = x[index], mean[index], var[index]
                    tmp = tmp[torch.randperm(tmp.size(0))].detach()
                    tmp = tmp * (var_tmp + self.eps).sqrt() + mean_tmp
                    x[index] = tmp
            else:
                # idx_swap = torch.randperm(N)
                x = x[idx_swap].detach()

                x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)
        return x


class AdaIN2d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta
        # return (1+gamma)*(x)+beta


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)


class Generator(nn.Module):
    # def __init__(self, n=16, kernelsize=3, imdim=3, imsize=[13, 13], zdim=10, device=0):
    def __init__(self, n=16, kernelsize=3, imdim=3, imsize=[13, 13], zdim=10):
        ''' w_ln 局部噪声权重
        '''
        super().__init__()
        stride = (kernelsize - 1) // 2
        self.zdim = zdim
        self.imdim = imdim
        self.imsize = imsize
        # self.device = device
        num_morph = 4

        self.conv1 = nn.Conv2d(imdim, 32, kernel_size=13)

        # self.conv2 = nn.Conv2d(imdim, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(imdim, 3, kernel_size=1)

        self.conv3 = nn.Conv2d(64 + 3, imdim, kernel_size=1)

        # self.conv4 = nn.Conv2d(64, 64, 1)

        self.FCLayer1 = nn.Sequential(
            nn.Linear(imdim, 64),
            nn.ReLU(),
        )

        self.FCLayer2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.FCLayer3 = nn.Sequential(
            nn.Linear(64, 32),
            # nn.Linear(64, 64), # 消融
            nn.ReLU(),
        )

        # self.speRandom = SpeRandomization(n)
        self.speRandom = SpeRandomization(64)

        self.speConvT = nn.ConvTranspose2d(64, 64, 13)

        self.spaRandom = SpaRandomization(3)

    #         self.localization = nn.Sequential(
    #             nn.Conv2d(3, 8, kernel_size=3, padding=1),
    #             nn.MaxPool2d(2),
    #             nn.ReLU(True),
    #             nn.Conv2d(8, 10, kernel_size=1),
    #             nn.MaxPool2d(2),
    #             nn.ReLU(True)
    #         )

    #         self.fc_loc = nn.Sequential(
    #             nn.Linear(10 * 3 * 3, 32),
    #             nn.ReLU(True),
    #             nn.Linear(32, 3 * 2)
    #         )
    #         # Initialize the weights/bias with identity transformation
    #         self.fc_loc[2].weight.data.zero_()
    #         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    #     def stn(self, x):
    #         xs = self.localization(x)
    #         xs = xs.view(-1, 10 * 3 * 3)
    #         theta = self.fc_loc(xs)
    #         theta = theta.view(-1, 2, 3)

    #         grid = F.affine_grid(theta, x.size(), align_corners=False)
    #         x = F.grid_sample(x, grid, align_corners=False)

    #         return x

    def forward(self, x, x_dct):
        bs, c, h, w = x.shape
        center = h // 2
        # spect2d = x[:, :, center:center+1, center:center+1].view()     # 102
        spect2d = x[:, :, center, center]  # 102

        spect1 = self.FCLayer1(spect2d)
        spect2 = self.FCLayer2(spect1)
        spect = self.FCLayer3(spect1 + spect2)  # .unsqueeze() reshape(bs, 32, 1, 1)  # bs,32,1,1

        spect = spect.unsqueeze(2).unsqueeze(3)
        x_dct = F.relu(self.conv1(x_dct))  # bs,32,1,1
        x_spect_dct = torch.cat((x_dct, spect), dim=1)  # bs,64,1,1
        # x_spect_dct = spect # 消融

        # x_spect_dct = self.conv4(torch.cat((x_dct, spect), dim=1))  # bs,64,1,1

        #
        x = F.relu(self.conv2(x))

        # idx = torch.randperm(x.shape[0])
        # x = self.stn(x)
        x, idx = self.spaRandom(x)

        # x, idx = self.spaRandom(x)    # 这个结果出现了Nan
        x_spect_dct = self.speRandom(x_spect_dct, idx)
        x_spect_dct = self.speConvT(x_spect_dct)  # bs,64,13,13
        x = torch.cat((x, x_spect_dct), dim=1)
        x = self.conv3(x)  # bs, 102, 13, 13

        return torch.sigmoid(x)