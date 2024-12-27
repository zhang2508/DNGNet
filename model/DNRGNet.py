# -*- coding: utf-8 -*-
"""
@author:
"""
import numpy as np
import torch
from einops import rearrange
from torch import nn, einsum
import torch.nn.functional as F
import math

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def knn(x, k):
    inner = -2 * torch.matmul(x.permute(0, 1, 3, 2), x)
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.permute(0, 1, 3, 2)
    idx = pairwise_distance.topk(k=k, dim=-1, sorted=False)[1]
    return idx

def get_graph_feature(x, k=2, idx=None):
    batch_size, n, feature, num_points = x.shape

    if idx is None:
        idx = knn(x, k=k)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_base = torch.arange(0, batch_size * n, device=device).view(batch_size, -1, 1, 1) * num_points
    idx = idx + idx_base

    _, _, num_dims, _ = x.size()

    x = x.permute(0, 1, 3, 2).contiguous()
    b = x.view(batch_size * n * num_points, -1)[idx, :]

    b = b.view(batch_size, n, num_points, k, num_dims)
    b = b.permute(0, 1, 4, 2, 3).contiguous()

    return b

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, window_size, k=20, pool_size=5):
        super().__init__()
        H, W = pair(image_size)
        self.window_size = window_size
        self.k = k

        self.topatch = nn.Sequential(
            nn.Unfold(kernel_size=window_size, stride=window_size, padding=window_size // 2)
        )
        self.backpatch = nn.Fold(output_size=(H, W), padding=window_size // 2,
                                 kernel_size=(window_size, window_size), stride=window_size)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(k, k),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)

    def forward(self, x, indx=None):
        out = self.topatch(x)
        out = rearrange(out, 'b (c nw nh) n ->b n c (nw nh)', nw=self.window_size, nh=self.window_size)
        out = get_graph_feature(out, self.k, indx)

        out = rearrange(out, 'b n c (nw nh) k -> (b k) (c nw nh) n', nw=self.window_size)
        out = self.backpatch(out)
        out = rearrange(out, '(b k) c nw nh ->b c nw nh k', k=self.k)

        out = self.fc(F.normalize(out, p=2, dim=1))
        out = out.max(dim=-1, keepdim=False)[0]
        out = self.conv(out)
        out = self.pool(out)

        return out

# 网络骨架
class basic_block(nn.Module):
    """基本残差块,由两层卷积构成"""
    def __init__(self, in_channels, out_channels, image_size, window_size, k=10, pool_size=5):
        super(basic_block, self).__init__()
        """
        :param in_channels: 输入通道
        :param out_channels:  输出通道
        """

        self.conv1 = Block(in_channels, out_channels, image_size, window_size, k, pool_size)
        self.conv2 = Block(out_channels, out_channels, image_size, window_size, k, pool_size)

        if in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        else:
            self.downsample = nn.Identity()

    def forward(self, inx, indx=None):
        x = self.conv1(inx, indx)
        x = self.conv2(x, indx)

        return x + self.downsample(inx)

class DNRGNet(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, window_size, layers=3, k=10, pool_size=5, num_classes=9):
        super().__init__()
        self.k = k
        self.window_size = window_size
        self.blockNums = [1, 1, 1, 1]
        self.out_channels = out_channels

        self.LinearProject = nn.Conv2d(in_channels, self.out_channels, 1)

        self.layer1 = basic_block(self.out_channels, 32, image_size, window_size, k, pool_size)
        self.layer2 = basic_block(32, 64, image_size, window_size, k, pool_size)
        # self.layer3 = basic_block(64, 128, image_size, window_size, k, pool_size)
        # self.layer4 = basic_block(128, 256, image_size, window_size, k, pool_size)

        self.cls = nn.Conv2d(64, num_classes, 1)

    def _make_layers(self, basicBlock, blockNum, channels, image_size, window_size, k=10, pool_size=5):
        """

        :param basicBlock: 基本残差块类
        :param blockNum: 当前层包含基本残差块的数目,resnet18每层均为2
        :param channels: 输出通道数
        :return:
        """
        layers = []
        for i in range(blockNum):
            if i == 0:
                layer = basicBlock(self.out_channels, channels, image_size, window_size, k, pool_size)
            else:
                layer = basicBlock(channels, channels, image_size, window_size, k, pool_size)
            layers.append(layer)
        self.out_channels = channels

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.LinearProject(x)
        indx = None

        x = self.layer1(x,indx)
        x = self.layer2(x,indx)
        # x = self.layer3(x,indx)
        # x = self.layer4(x,indx)

        return self.cls(x)

if __name__ == "__main__":
    num_PC = 3
    classnum = 16
    image_size = (145, 145)
    net = DNRGNet(in_channels=3,
                out_channels=16,
                image_size=(145, 145),
                window_size=7,
                layers=2,
                k=3,
                pool_size=7,
                num_classes=16).cuda()

    from torchsummary import summary

    summary(net, (num_PC, 145, 145), device="cuda")

