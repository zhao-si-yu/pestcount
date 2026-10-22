"""统一推理接口：训练脚本、评估脚本、GUI 与命令行共用。

对外只暴露两件事：加载权重、给一张 RGB 图返回计数与密度图。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .model import build_model
from .utils import to_input_tensor

__all__ = ["Predictor", "load_checkpoint"]


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> Dict[str, Any]:
    """兼容不同 torch 版本的 checkpoint 读取。"""
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:          # torch < 1.13 没有 weights_only 参数
        return torch.load(str(path), map_location=device)


class Predictor:
    """密度估计推理封装。

    Args:
        ckpt: 权重路径；为 None 时随机初始化（仅用于流程联调）。
        device: "cuda" / "cpu"，自动回退。
        infer_size: 推理时长边缩放尺寸，需与训练时一致。
        width: 网络宽度，会从 checkpoint 里自动读取，显式传入可覆盖。
    """

    def __init__(self, ckpt: Optional[str | Path] = None,
                 device: str = "cuda", infer_size: Optional[int] = None,
                 width: Optional[float] = None):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        cfg: Dict[str, Any] = {}
        state = None
        if ckpt is not None:
            ckpt_path = Path(ckpt)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"权重不存在: {ckpt_path}")
            blob = load_checkpoint(ckpt_path, self.device)
            state = blob.get("model", blob) if isinstance(blob, dict) else blob
            cfg = blob.get("args", {}) if isinstance(blob, dict) else {}

        self.width = float(width if width is not None else cfg.get("width", 1.5))
        self.infer_size = int(infer_size if infer_size is not None
                              else cfg.get("infer_size", 1024))
        self.stride = 4

        self.model = build_model(width=self.width).to(self.device).eval()
        if state is not None:
            self.model.load_state_dict(state)

    @torch.no_grad()
    def predict(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        """对单张 RGB 图推理。

        Returns:
            {"count": float, "density": (h/4, w/4) ndarray, "size": (h, w)}
        """
        if img_rgb.ndim == 2:
            img_rgb = np.stack([img_rgb] * 3, axis=-1)
        tensor, (h, w) = to_input_tensor(img_rgb, infer_size=self.infer_size,
                                         stride=self.stride)
        tensor = tensor.to(self.device)
        pred = self.model(tensor)
        density = pred[0, 0].detach().cpu().numpy()
        return {"count": float(density.sum()), "density": density, "size": (h, w)}

    @torch.no_grad()
    def predict_batch(self, images, batch_size: int = 4):
        """批量推理（尺寸会统一到 infer_size，可堆叠）。"""
        results = []
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            tensors, sizes = [], []
            for img in chunk:
                t, s = to_input_tensor(img, infer_size=self.infer_size, stride=self.stride)
                tensors.append(t)
                sizes.append(s)
            batch = torch.cat(tensors, dim=0).to(self.device)
            preds = self.model(batch).detach().cpu().numpy()
            for j, (p, s) in enumerate(zip(preds, sizes)):
                d = p[0]
                results.append({"count": float(d.sum()), "density": d, "size": s,
                                "index": i + j})
        return results
