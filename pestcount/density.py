"""密度图生成。

点监督计数任务的核心是把「点标注」转换成「密度图」作为回归目标。
本模块提供几何自适应高斯核（geometry-adaptive kernel）：

    对第 i 个点，取其 K 个最近邻的平均欧氏距离 d_i，
    令核宽 sigma_i = beta * d_i (beta 可调，默认 0.3)。

为什么不用固定 sigma：
    诱捕板照片存在明显透视，近处虫体间隔大、远处虫体挤在一起。
    固定 sigma 会让密集区的高斯过度叠加（密度峰值虚高）、
    稀疏区的高斯过于尖锐（积分不守恒），最终导致计数出现
    随密度变化的系统性偏差。自适应核让每个点的核宽随局部
    密度变化，密度图的积分严格等于点数。
"""

from __future__ import annotations

import numpy as np

__all__ = ["geometry_adaptive_sigma", "points_to_density", "density_to_points_count"]


def geometry_adaptive_sigma(points: np.ndarray, k: int = 3, beta: float = 0.3,
                            min_sigma: float = 1.0, max_sigma: float = 100.0) -> np.ndarray:
    """为每个点计算几何自适应核宽。

    Args:
        points: (N, 2) 坐标数组，格式为 (x, y)。
        k: 近邻数量。点数不足 k+1 时退化为全局平均距离。
        beta: 核宽系数，sigma = beta * 平均近邻距离。
        min_sigma / max_sigma: 核宽裁剪范围，防止极端值破坏数值稳定性。

    Returns:
        (N,) 的 sigma 数组。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    if n == 1:
        return np.array([max(min_sigma, 4.0 * beta)], dtype=np.float64)

    # 两两距离矩阵：N 通常小于 2000，暴力计算足够快且不引入 scipy 依赖
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    # 排除自身后取前 k 个近邻；k 超过可用邻居数时截断
    np.fill_diagonal(dist, np.inf)
    k_eff = min(k, n - 1)
    knn = np.partition(dist, k_eff - 1, axis=1)[:, :k_eff]
    mean_dist = knn.mean(axis=1)

    sigma = beta * mean_dist
    # 处理重合点：mean_dist 可能为 0
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, min_sigma)
    return np.clip(sigma, min_sigma, max_sigma)


def points_to_density(points: np.ndarray, height: int, width: int,
                      k: int = 3, beta: float = 0.3,
                      min_sigma: float = 1.0, max_sigma: float = 100.0,
                      truncate: float = 3.0) -> np.ndarray:
    """把点集渲染成密度图，且保证 density.sum() == len(points)。

    Args:
        points: (N, 2)，(x, y)，坐标落在 [0, width) × [0, height)。
        height, width: 输出密度图尺寸。
        truncate: 高斯窗截断半径，单位为 sigma。

    Returns:
        (height, width) float32 密度图。
    """
    density = np.zeros((height, width), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return density.astype(np.float32)

    sigmas = geometry_adaptive_sigma(pts, k=k, beta=beta,
                                     min_sigma=min_sigma, max_sigma=max_sigma)

    # 预计算归一化网格（每个点窗口大小不同，用局部切片累加）
    for (x, y), sigma in zip(pts, sigmas):
        radius = max(1, int(round(truncate * sigma)))
        x0, x1 = int(np.floor(x)) - radius, int(np.ceil(x)) + radius + 1
        y0, y1 = int(np.floor(y)) - radius, int(np.ceil(y)) + radius + 1

        # 裁剪到图像范围内，同时记录有效切片
        gx0, gx1 = max(0, x0), min(width, x1)
        gy0, gy1 = max(0, y0), min(height, y1)
        if gx0 >= gx1 or gy0 >= gy1:
            continue

        xs = np.arange(gx0, gx1, dtype=np.float64) - x
        ys = np.arange(gy0, gy1, dtype=np.float64) - y
        gauss = np.exp(-(xs[None, :] ** 2 + ys[:, None] ** 2) / (2.0 * sigma ** 2))
        density[gy0:gy1, gx0:gx1] += gauss

    # 截断与边界裁剪会损失少量质量，这里做一次全局归一化，
    # 保证密度图积分严格等于真实点数（计数任务的硬约束）
    total = density.sum()
    if total > 0:
        density *= pts.shape[0] / total
    return density.astype(np.float32)


def density_to_points_count(density: np.ndarray) -> float:
    """密度图积分即预测计数。"""
    return float(np.asarray(density, dtype=np.float64).sum())
