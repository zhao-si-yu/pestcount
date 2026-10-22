"""训练损失组合。

两个分项：

1. density L1：像素级密度图回归，负责「虫在哪」；
2. 局部计数一致性（local count consistency）：把预测与真值密度图
   都划分成 g×g 网格并求每格积分，对每格的「计数」做 L1。

为什么需要第 2 项：
   纯密度图回归在数值上容易收敛到「峰值偏低 + 底盘变宽」的解——
   密度图视觉上贴合，但积分出来的总数系统性偏小。局部计数项直接
   约束网格计数，等价于给网络加了「积分守恒」的硬监督，在稀疏
   目标（单板只有十几只虫）场景下对 MAE 的收益最明显。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DensityCountLoss"]


def _grid_sum(x: torch.Tensor, grid: int) -> torch.Tensor:
    """把 (B,1,H,W) 划分为 grid×grid 并返回每格的积分值。"""
    h, w = x.shape[-2:]
    pooled = F.adaptive_avg_pool2d(x, (grid, grid))
    area = (h * w) / (grid * grid)
    return pooled * area


class DensityCountLoss(nn.Module):
    """密度图 L1 + 局部计数一致性。

    Args:
        lambda_local: 局部计数项权重。
        grid: 网格划分数。
        lambda_global: 全局计数项权重（对总数做相对误差约束，可选）。
    """

    def __init__(self, lambda_local: float = 1.0, grid: int = 4,
                 lambda_global: float = 0.1):
        super().__init__()
        self.lambda_local = lambda_local
        self.grid = grid
        self.lambda_global = lambda_global

    def forward(self, pred: torch.Tensor, gt: torch.Tensor
                ) -> tuple[torch.Tensor, dict[str, float]]:
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear",
                                 align_corners=False)

        l_density = F.l1_loss(pred, gt)
        l_local = F.l1_loss(_grid_sum(pred, self.grid), _grid_sum(gt, self.grid))

        pc, gc = pred.sum(dim=(1, 2, 3)), gt.sum(dim=(1, 2, 3))
        l_global = ((pc - gc).abs() / (gc + 1.0)).mean()

        total = l_density + self.lambda_local * l_local + self.lambda_global * l_global
        return total, {
            "loss": float(total.detach()),
            "l_density": float(l_density.detach()),
            "l_local": float(l_local.detach()),
            "l_global": float(l_global.detach()),
        }
