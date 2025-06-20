#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CauseHSI 
@File    ：hsic.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2025/6/20 16:16 
"""
import torch
import numpy as np


def pairwise_distances(x):
  x_distances = torch.sum(x**2,-1).reshape((-1,1))
  return -2*torch.mm(x,x.t()) + x_distances + x_distances.t()

def kernelMatrixGaussian(x, sigma=1):

    pairwise_distances_ = pairwise_distances(x)
    gamma = -1.0 / (sigma ** 2)
    return torch.exp(gamma * pairwise_distances_)

def kernelMatrixLinear(x):
  return torch.matmul(x,x.t())

# check
def HSIC(X, Y, kernelX="Linear", kernelY="Linear", sigmaX=1, sigmaY=1,
         log_median_pairwise_distance=False):
  m,_ = X.shape
  device = X.device
  assert(m>1)

  K = kernelMatrixGaussian(X,sigmaX) if kernelX == "Gaussian" else kernelMatrixLinear(X)
  L = kernelMatrixGaussian(Y,sigmaY) if kernelY == "Gaussian" else kernelMatrixLinear(Y)

  H = torch.eye(m, device=device) - 1.0/m * torch.ones((m,m), device=device)
  H = H.float().to(device)#.cuda()

  Kc = torch.mm(H,torch.mm(K,H))

  HSIC = torch.trace(torch.mm(L,Kc))/((m-1)**2)
  return HSIC


def median_pairwise_distance(X):
    t = pairwise_distances(X).detach()
    triu_indices = t.triu(diagonal=1).nonzero().T

    if triu_indices[0].shape[0] == 0 or triu_indices[1].shape[0] == 0:
        return 0.
    else:
        return torch.median(t[triu_indices[0], triu_indices[1]]).item()