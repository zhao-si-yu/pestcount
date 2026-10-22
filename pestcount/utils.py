"""通用工具：图像读写、随机种子、推理预处理、可视化叠加。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

__all__ = [
    "imread_unicode", "imwrite_unicode", "set_seed", "to_input_tensor",
    "heatmap_overlay", "denormalize", "draw_points",
]


def imread_unicode(path: str | Path) -> Optional[np.ndarray]:
    """支持中文/空格路径的读图（cv2.imread 在 Windows 下会因非 ASCII 路径失败）。"""
    path = Path(path)
    if not path.exists():
        return None
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def imwrite_unicode(path: str | Path, img: np.ndarray, quality: int = 95) -> bool:
    """支持中文路径的写图（quality 仅对 jpg 生效）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_input_tensor(img_rgb: np.ndarray, infer_size: int = 1024,
                    stride: int = 4) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """把任意尺寸 RGB 图缩放到推理尺寸并归一化。

    Returns:
        (1,3,H,W) tensor 与缩放后尺寸 (h, w)。
    """
    h, w = img_rgb.shape[:2]
    s = infer_size / max(h, w)
    unit = stride * 4
    nh = max(unit, int(round(h * s / unit)) * unit)
    nw = max(unit, int(round(w * s / unit)) * unit)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, (nh, nw)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """把归一化后的 (3,H,W) tensor 还原成 uint8 RGB。"""
    x = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def heatmap_overlay(img_rgb: np.ndarray, density: np.ndarray,
                    alpha: float = 0.55) -> np.ndarray:
    """把密度图以 JET 热力图形式叠加到原图上（返回 BGR，便于 cv2 保存）。"""
    h, w = img_rgb.shape[:2]
    d = cv2.resize(density.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    # 用 99.5 分位数做归一化，避免个别峰值把整张热力图压黑
    vmax = float(np.percentile(d, 99.5)) if d.max() > 0 else 1.0
    vmax = max(vmax, 1e-6)
    dn = np.clip(d / vmax, 0, 1)
    heat = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_JET)

    base = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # 只在有响应的地方上色，背景保持原样，视觉上更干净
    mask = (dn > 0.05).astype(np.float32)[..., None]
    out = base * (1 - alpha * mask) + heat * (alpha * mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_points(img_bgr: np.ndarray, points: np.ndarray, radius: int = 4,
                color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    out = img_bgr.copy()
    for x, y in np.asarray(points).reshape(-1, 2):
        cv2.circle(out, (int(round(x)), int(round(y))), radius, color, -1, cv2.LINE_AA)
    return out
