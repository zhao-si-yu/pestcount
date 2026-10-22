"""命令行批量计数（无 GUI，便于脚本化/服务器使用）。

用法::

    python tools/count_cli.py --ckpt checkpoints/best.pt --input data/images/boards \
        --out result.csv --vis out/vis
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pestcount.predictor import Predictor
from pestcount.utils import heatmap_overlay, imread_unicode, imwrite_unicode

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--input", type=str, required=True, help="图片或目录")
    ap.add_argument("--out", type=str, default="result.csv")
    ap.add_argument("--vis", type=str, default="", help="热力图输出目录（可选）")
    ap.add_argument("--infer-size", type=int, default=0, help="0 表示用权重自带配置")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    pred = Predictor(args.ckpt, device=args.device,
                     infer_size=args.infer_size or None)

    src = Path(args.input)
    files = [src] if src.is_file() else sorted(
        [p for p in src.rglob("*") if p.suffix.lower() in EXTS])
    if not files:
        raise SystemExit(f"没有找到图片：{src}")

    vis_dir = Path(args.vis) if args.vis else None
    rows, total = [], 0.0
    for i, path in enumerate(files):
        img = imread_unicode(path)
        if img is None:
            print(f"[skip] {path}")
            continue
        t0 = time.perf_counter()
        out = pred.predict(img[:, :, ::-1])
        dt = (time.perf_counter() - t0) * 1000
        rows.append([path.name, f"{out['count']:.2f}", f"{dt:.1f}"])
        total += out["count"]
        print(f"{i+1:4d}/{len(files)}  {path.name:<32} count={out['count']:8.2f}  {dt:6.1f}ms")
        if vis_dir is not None:
            imwrite_unicode(vis_dir / f"{path.stem}_heat.jpg",
                            heatmap_overlay(img[:, :, ::-1], out["density"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "predicted_count", "latency_ms"])
        w.writerows(rows)
        w.writerow(["TOTAL", f"{total:.2f}", ""])
    print(f"\n共 {len(rows)} 张，合计 {total:.0f} 只  ->  {out_path}")


if __name__ == "__main__":
    main()
