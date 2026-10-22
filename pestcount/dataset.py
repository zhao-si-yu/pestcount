"""点标注数据集。

标注格式（JSON，一个 split 一个文件）::

    {
      "meta": {"beta": 0.3, "k": 3, "stride": 4},
      "items": [
        {"file_name": "images/board_0001.jpg",
         "points": [[120.5, 88.0], [301.2, 402.7]],
         "width": 1920, "height": 1440}
      ]
    }

约定：
  * points 为虫体中心像素坐标 (x, y)，x 对应列、y 对应行；
  * 密度图标在 1/4 分辨率上生成（stride=4），因此点坐标与核宽同步除以 4；
  * 一个没有任何点的空样本是合法样本（负样本），用来抑制误检。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .density import points_to_density
from .utils import imread_unicode

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_index(json_path: str | Path) -> Dict[str, Any]:
    """读取标注索引文件。"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_index(items: Sequence[Dict[str, Any]], json_path: str | Path,
               meta: Optional[Dict[str, Any]] = None) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {"beta": 0.3, "k": 3, "stride": 4},
               "items": list(items)}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _augment_color(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """颜色抖动：模拟不同光照、白平衡与曝光。"""
    if rng.random() < 0.5:                       # 亮度
        img = img * rng.uniform(0.75, 1.25)
    if rng.random() < 0.5:                       # 对比度
        mean = img.mean()
        img = (img - mean) * rng.uniform(0.8, 1.2) + mean
    if rng.random() < 0.3:                       # 饱和度 / 通道偏移
        shift = np.array([rng.uniform(-12, 12) for _ in range(3)], dtype=np.float32)
        img = img + shift[None, None, :]
    if rng.random() < 0.2:                       # 轻微模糊（失焦）
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.2:                       # 噪声（高 ISO）
        img = img + np.random.normal(0, rng.uniform(2, 8), img.shape).astype(np.float32)
    return np.clip(img, 0, 255)


class PointCountDataset(Dataset):
    """点监督计数数据集。

    Args:
        items: 索引文件中的 items 列表。
        root: 图像根目录（file_name 相对于此目录）。
        mode: "train" 随机缩放裁剪 + 增强；"val" 整图缩放到 infer_size。
        crop_size: 训练裁剪边长（需能被 stride*4 整除，默认 512）。
        scale_range: 训练随机缩放范围。
        stride: 密度图相对输入的下采样倍率，需与模型 out_stride 一致。
        infer_size: 验证/推理时长边缩放尺寸。
    """

    def __init__(self, items: List[Dict[str, Any]], root: str | Path,
                 mode: str = "train", crop_size: int = 512,
                 scale_range: Tuple[float, float] = (0.5, 1.0),
                 stride: int = 4, infer_size: int = 1024,
                 beta: float = 0.3, k: int = 3):
        assert mode in ("train", "val")
        self.items = items
        self.root = Path(root)
        self.mode = mode
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.stride = stride
        self.infer_size = infer_size
        self.beta = beta
        self.k = k

    def __len__(self) -> int:
        return len(self.items)

    def _make_density(self, points: np.ndarray, h: int, w: int) -> np.ndarray:
        """在 1/stride 分辨率上生成密度图，点坐标与核宽同步缩放。"""
        hh, ww = h // self.stride, w // self.stride
        if points.size == 0:
            return np.zeros((hh, ww), dtype=np.float32)
        scaled = points.astype(np.float64) / self.stride
        return points_to_density(
            scaled, hh, ww, k=self.k, beta=self.beta,
            min_sigma=1.0, max_sigma=max(hh, ww) / 4.0,
        )

    def _random_crop(self, img: np.ndarray, points: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        # 图像（随机缩放后）小于裁剪尺寸时先放大，保证 batch 内尺寸一致
        if min(h, w) < self.crop_size:
            s = self.crop_size / min(h, w)
            nh, nw = int(round(h * s)), int(round(w * s))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            if points.size:
                points = points * np.array([nw / w, nh / h], dtype=np.float64)
            h, w = nh, nw
        c = self.crop_size
        # 50% 概率围绕一个随机点裁剪，保证正样本里确实有虫体，
        # 另外 50% 纯随机，保留空背景作为负样本
        if points.shape[0] > 0 and random.random() < 0.5:
            cx, cy = points[random.randrange(len(points))]
            x0 = int(np.clip(cx - c / 2 + random.uniform(-c / 4, c / 4), 0, w - c))
            y0 = int(np.clip(cy - c / 2 + random.uniform(-c / 4, c / 4), 0, h - c))
        else:
            x0 = random.randint(0, max(0, w - c))
            y0 = random.randint(0, max(0, h - c))
        patch = img[y0:y0 + c, x0:x0 + c]
        if points.shape[0]:
            sel = points[(points[:, 0] >= x0) & (points[:, 0] < x0 + c) &
                         (points[:, 1] >= y0) & (points[:, 1] < y0 + c)]
            sel = sel - np.array([x0, y0], dtype=np.float64)
        else:
            sel = points
        return np.ascontiguousarray(patch), sel

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        img_path = self.root / item["file_name"]
        img = imread_unicode(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        points = np.asarray(item.get("points", []), dtype=np.float64).reshape(-1, 2)

        if self.mode == "train":
            s = random.uniform(*self.scale_range)
            if abs(s - 1.0) > 1e-3:
                nh, nw = max(1, int(round(img.shape[0] * s))), max(1, int(round(img.shape[1] * s)))
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
                points = points * s
            img, points = self._random_crop(img, points)
            # 裁剪边长对齐到 stride*4，避免密度图尺寸取整漂移
            c = img.shape[0] - img.shape[0] % (self.stride * 4)
            img, points = img[:c, :c], points
            if random.random() < 0.5:                      # 水平翻转
                img = np.ascontiguousarray(img[:, ::-1])
                points = np.stack([c - 1 - points[:, 0], points[:, 1]], axis=1) \
                    if points.size else points
            img = _augment_color(img, random)
        else:
            h, w = img.shape[:2]
            s = self.infer_size / max(h, w)
            nh = max(self.stride * 4, int(round(h * s / (self.stride * 4))) * self.stride * 4)
            nw = max(self.stride * 4, int(round(w * s / (self.stride * 4))) * self.stride * 4)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            points = points * np.array([nw / w, nh / h], dtype=np.float64)

        img = np.clip(img, 0, 255) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()

        density = self._make_density(points, tensor.shape[1], tensor.shape[2])
        return {
            "image": tensor,
            "density": torch.from_numpy(density).unsqueeze(0).float(),
            "count": float(points.shape[0]),
            "name": item["file_name"],
            "points": torch.from_numpy(points.astype(np.float32)).reshape(-1, 2),
        }


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """训练用 collate：图像与密度图尺寸一致时直接堆叠，否则返回列表。"""
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "density": torch.stack([b["density"] for b in batch], dim=0),
        "count": torch.tensor([b["count"] for b in batch], dtype=torch.float32),
        "name": [b["name"] for b in batch],
        "points": [b["points"] for b in batch],
    }
