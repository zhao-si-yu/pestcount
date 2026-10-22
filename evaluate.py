"""评估入口：输出计数指标并保存可视化对比图。

用法::

    python evaluate.py --ckpt checkpoints/best.pt \
                       --json data/annotations/val.json \
                       --data-root data/images --vis-dir docs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from pestcount.dataset import PointCountDataset, load_index
from pestcount.metrics import count_metrics, summarize
from pestcount.predictor import Predictor
from pestcount.utils import (denormalize, draw_points, heatmap_overlay,
                             imwrite_unicode, set_seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    p.add_argument("--json", type=str, default="data/annotations/val.json")
    p.add_argument("--data-root", type=str, default="data/images")
    p.add_argument("--vis-dir", type=str, default="docs/figures")
    p.add_argument("--max-vis", type=int, default=8)
    p.add_argument("--report", type=str, default="checkpoints/eval_report.json")
    return p.parse_args()


def save_visual(img_rgb: np.ndarray, gt_points: np.ndarray, density: np.ndarray,
                pred_count: float, gt_count: float, out_path: Path) -> None:
    """三联图：原图+真值点 / 预测热力图 / 并排对比。"""
    overlay = heatmap_overlay(img_rgb, density)
    gt_vis = draw_points(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), gt_points,
                         radius=max(2, img_rgb.shape[1] // 400))
    h = 480
    panels = []
    for im, title in ((gt_vis, f"GT: {int(gt_count)}"),
                      (overlay, f"Pred: {pred_count:.1f}")):
        s = h / im.shape[0]
        im = cv2.resize(im, (int(im.shape[1] * s), h))
        cv2.putText(im, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(im, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(im)
    imwrite_unicode(out_path, np.hstack(panels))


def main() -> None:
    args = parse_args()
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = Predictor(args.ckpt, device=device)

    items = load_index(args.json)["items"]
    ds = PointCountDataset(items, args.data_root, mode="val",
                           infer_size=predictor.infer_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0])

    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    preds, gts, details = [], [], []
    for i, sample in enumerate(loader):
        img_rgb = denormalize(sample["image"])
        out = predictor.predict(img_rgb)
        preds.append(out["count"])
        gts.append(sample["count"])
        details.append({"name": sample["name"], "pred": out["count"],
                        "gt": sample["count"], "abs_err": abs(out["count"] - sample["count"])})
        if i < args.max_vis:
            save_visual(img_rgb, sample["points"].numpy(), out["density"],
                        out["count"], sample["count"],
                        vis_dir / f"eval_{Path(sample['name']).stem}.jpg")

    metrics = count_metrics(preds, gts)
    print("RESULT:", summarize(metrics))

    report = {"metrics": metrics,
              "params": sum(p.numel() for p in predictor.model.parameters()),
              "items": details}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"report -> {args.report}")
    print(f"visuals -> {vis_dir}")


if __name__ == "__main__":
    main()
