"""核心逻辑自检（无需数据集，可直接 python tests/test_core.py 运行）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pestcount.density import geometry_adaptive_sigma, points_to_density
from pestcount.loss import DensityCountLoss
from pestcount.metrics import count_metrics
from pestcount.model import build_model, count_parameters


def test_density_conservation() -> None:
    rng = np.random.default_rng(0)
    # 混入一处高度密集的点簇，检验自适应核在密集区是否仍然积分守恒
    sparse = rng.uniform([0, 0], [512, 512], size=(20, 2))
    dense = rng.normal([300, 300], 8, size=(30, 2))
    pts = np.vstack([sparse, dense])

    d = points_to_density(pts, 512, 512, k=3, beta=0.3)
    assert d.shape == (512, 512), d.shape
    assert d.min() >= 0
    err = abs(float(d.sum()) - len(pts))
    assert err < 1e-3, f"积分不守恒，误差 {err}"
    print(f"[ok] 密度图积分守恒: sum={d.sum():.4f} vs N={len(pts)}")


def test_adaptive_sigma_responds_to_density() -> None:
    # 密集簇内的点，其核宽应显著小于稀疏区域
    sparse = np.array([[20.0, 20.0], [480.0, 480.0]])
    dense = np.array([[250.0, 250.0], [255.0, 250.0], [250.0, 255.0], [255.0, 255.0]])
    pts = np.vstack([sparse, dense])
    sig = geometry_adaptive_sigma(pts, k=3, beta=0.3)
    print(f"[ok] 自适应核宽  稀疏点 sigma={sig[:2].mean():.3f}  "
          f"密集点 sigma={sig[2:].mean():.3f}")
    assert sig[:2].mean() > sig[2:].mean() * 3


def test_model_shape_and_size() -> None:
    model = build_model()
    n = count_parameters(model)
    x = torch.randn(2, 3, 512, 512)
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 128, 128), y.shape
    assert (y >= 0).all(), "密度图不应出现负值"
    print(f"[ok] 模型输出 {tuple(y.shape)}  参数量 {n/1e6:.3f}M")
    assert n < 1_500_000, "参数量超出轻量化目标"


def test_loss_and_metrics() -> None:
    crit = DensityCountLoss()
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 512, size=(12, 2))
    gt = torch.from_numpy(points_to_density(pts / 4, 128, 128)).unsqueeze(0).unsqueeze(0)
    pred = gt * 0.8 + 0.001          # 模拟系统性低估 20%
    loss, parts = crit(pred, gt)
    print(f"[ok] 损失 {float(loss):.4f} 明细 {parts}")

    m = count_metrics([10.0, 20.0, 5.0], [12.0, 18.0, 5.0])
    print(f"[ok] 指标 MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} Acc={m['acc']:.3f}")


if __name__ == "__main__":
    test_density_conservation()
    test_adaptive_sigma_responds_to_density()
    test_model_shape_and_size()
    test_loss_and_metrics()
    print("\n全部自检通过")
