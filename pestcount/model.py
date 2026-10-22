"""轻量密度估计网络。

设计约束（面向嵌入式/端侧部署）：
  * 参数量 < 1M，便于 INT8 量化后塞进边缘盒子；
  * 全程使用 GroupNorm 而非 BatchNorm —— 训练 batch 小（8~16）
    时 BN 的统计量不稳，而 GN 与 batch 无关，量化后也更友好；
  * 深度可分离卷积压缩计算量；
  * 输出 1/4 分辨率的单通道密度图，上采样只做一次倍数为 4 的插值。

结构：
    stem  (s2)  -> 1/2
    stage1(s2)  -> 1/4
    stage2(s2)  -> 1/8
    stage3(s2)  -> 1/16
    up + fuse   -> 1/8
    up + fuse   -> 1/4
    head 1x1    -> 1 channel
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LiteCountNet", "build_model", "count_parameters"]


def _norm(channels: int, groups: int = 8) -> nn.Module:
    g = groups
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


class ConvBNAct(nn.Sequential):
    """标准 3x3 卷积块。"""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            _norm(cout),
            nn.SiLU(inplace=True),
        )


class DWBlock(nn.Sequential):
    """深度可分离卷积块：depthwise 3x3 + pointwise 1x1。"""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
            _norm(cin),
            nn.SiLU(inplace=True),
            nn.Conv2d(cin, cout, 1, 1, 0, bias=False),
            _norm(cout),
            nn.SiLU(inplace=True),
        )


class UpsampleFuse(nn.Module):
    """双线性上采样 + 与同级特征相加 + 卷积平滑。"""

    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.lateral = nn.Conv2d(cskip, cin, 1, 1, 0, bias=False) if cskip != cin else nn.Identity()
        self.fuse = ConvBNAct(cin, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(x + self.lateral(skip))


class LiteCountNet(nn.Module):
    """轻量密度图回归网络。"""

    def __init__(self, width: float = 1.0, in_channels: int = 3, head_init: float = -8.0):
        super().__init__()
        w = [max(8, int(round(c * width))) for c in (24, 48, 96, 128, 64)]

        self.stem = ConvBNAct(in_channels, w[0], stride=2)      # 1/2
        self.stage1 = DWBlock(w[0], w[1], stride=2)             # 1/4
        self.stage2 = DWBlock(w[1], w[2], stride=2)             # 1/8
        self.stage3 = DWBlock(w[2], w[3], stride=2)             # 1/16

        self.up3 = UpsampleFuse(w[3], w[2], w[4])               # 1/8
        self.up2 = UpsampleFuse(w[4], w[1], w[4])               # 1/4

        self.head = nn.Conv2d(w[4], 1, 1, 1, 0)
        # 初始权重置零、偏置取负值：训练初期输出恒为 softplus(-8)≈3e-4 的
        # 常数场，接近「零密度」解，避免随机初始化时密度图直接爆量导致
        # 前几个 epoch 梯度爆炸（软木板类图像里虫体像素占比极低）
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, head_init)
        self.out_stride = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        f = self.stem(x)
        c1 = self.stage1(f)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)

        d = self.up3(c3, c2)
        d = self.up2(d, c1)
        # softplus 保证密度图非负（计数任务不允许负密度）
        out = F.softplus(self.head(F.silu(d)))

        target = (h // self.out_stride, w // self.out_stride)
        if out.shape[-2:] != target:
            out = F.interpolate(out, size=target, mode="bilinear", align_corners=False)
        return out


def build_model(width: float = 1.5) -> LiteCountNet:
    """默认 width=1.5（约 0.35M 参数），在容量与端侧体积之间取平衡。

    经验上 width 低于 1.2 时对密集小目标（<10px）欠拟合明显；
    高于 2.0 后 MAE 收益递减而延迟线性上升。
    """
    return LiteCountNet(width=width)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
