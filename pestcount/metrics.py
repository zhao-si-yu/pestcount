"""计数任务的评价指标。

注意：计数任务不使用 mAP / IoU。点标注没有框，目标之间高度重叠，
检测指标在该任务上既不稳定也不可比。这里统一采用计数回归指标：

    MAE  = mean(|pred - gt|)                      平均绝对计数误差（主指标）
    RMSE = sqrt(mean((pred - gt)^2))              对大误差更敏感
    Acc  = mean(1 - |pred-gt| / max(gt, 1))       计数准确率（便于直观理解）
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = ["count_metrics", "summarize"]


def count_metrics(preds: Sequence[float], gts: Sequence[float]) -> Dict[str, float]:
    p = np.asarray(preds, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    if p.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "acc": 0.0, "bias": 0.0, "n": 0}

    err = p - g
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    acc = float(np.mean(1.0 - np.abs(err) / np.maximum(g, 1.0)))
    bias = float(np.mean(err))       # 正值代表系统性高估
    return {"mae": mae, "rmse": rmse, "acc": acc, "bias": bias, "n": int(p.size)}


def summarize(metrics: Dict[str, float]) -> str:
    return (f"MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
            f"Acc={metrics['acc'] * 100:.2f}%  Bias={metrics['bias']:+.3f}  "
            f"(n={metrics['n']})")
