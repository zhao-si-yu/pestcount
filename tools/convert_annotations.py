"""把公开数据集的框标注转换成本项目的点标注格式。

支持四种常见格式，一次转换即可直接喂给 train.py：

    voc         PASCAL VOC XML（Yellow Sticky Traps 等，每图一个 .xml）
    coco        COCO instances JSON（单文件）
    yolo        YOLO txt（每图一个 .txt，类别 cx cy w h 归一化）
    pst-csv     Pest Sticky Traps 的 CSV（imageName,X,Y,class，本身就是点标注）

用法::

    # VAT 格式（黄板粘虫数据集）
    python tools/convert_annotations.py --format voc \
        --images data/raw/yellow-sticky/images \
        --ann data/raw/yellow-sticky/annotations \
        --out data/images/boards --out-json data/annotations/yellow_sticky.json

    # PST 点标注（原生就是点，直接复制）
    python tools/convert_annotations.py --format pst-csv \
        --images data/raw/pst/train --ann data/raw/pst/train/annotations.csv \
        --out data/images/pst --out-json data/annotations/pst.json

关于「框中心当作点」的取舍：
    密度估计需要的是虫体中心。对于小目标（框短边 < 15px）框中心与真实
    中心误差通常 < 1px，可以接受；对于弯曲/细长的虫体，框中心会偏移到
    虫体以外，这时建议 --min-size / --max-size 过滤掉过大的框，或改用
    标注工具重新点。脚本会把过滤情况打印出来，请不要忽略。
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def read_size(path: Path) -> tuple[int, int]:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    return w, h


def parse_voc(path: Path) -> list[tuple[float, float, str]]:
    """返回 [(cx, cy, class_name)]。"""
    out = []
    root = ET.parse(path).getroot()
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "insect").strip()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x0 = float(bb.findtext("xmin", "0"))
        y0 = float(bb.findtext("ymin", "0"))
        x1 = float(bb.findtext("xmax", "0"))
        y1 = float(bb.findtext("ymax", "0"))
        out.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0, name))
    return out


def convert_voc(images: list[Path], ann_dir: Path, args) -> tuple[list, Counter]:
    items, stats = [], Counter()
    for img_path in images:
        xml_path = ann_dir / (img_path.stem + ".xml")
        if not xml_path.exists():
            stats["missing_xml"] += 1
            continue
        w, h = read_size(img_path)
        pts = [(x, y) for x, y, _ in parse_voc(xml_path)]
        items.append({"src": img_path, "points": pts, "width": w, "height": h})
        stats["images"] += 1
        stats["points"] += len(pts)
    return items, stats


def convert_coco(images: list[Path], ann_file: Path, args) -> tuple[list, Counter]:
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_id = {im["id"]: im for im in data.get("images", [])}
    pts_by_img: dict[int, list] = {}
    for an in data.get("annotations", []):
        x, y, bw, bh = an["bbox"]
        pts_by_img.setdefault(an["image_id"], []).append((x + bw / 2, y + bh / 2))

    items, stats = [], Counter()
    for img_path in images:
        rec = next((v for v in by_id.values()
                    if Path(v.get("file_name", "")).name == img_path.name), None)
        if rec is None:
            stats["missing_ann"] += 1
            continue
        pts = pts_by_img.get(rec["id"], [])
        items.append({"src": img_path, "points": pts,
                      "width": rec.get("width", 0), "height": rec.get("height", 0)})
        stats["images"] += 1
        stats["points"] += len(pts)
    return items, stats


def convert_yolo(images: list[Path], label_dir: Path, args) -> tuple[list, Counter]:
    items, stats = [], Counter()
    for img_path in images:
        txt = label_dir / (img_path.stem + ".txt")
        if not txt.exists():
            stats["missing_label"] += 1
            continue
        w, h = read_size(img_path)
        pts = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy = float(parts[1]) * w, float(parts[2]) * h
            pts.append((cx, cy))
        items.append({"src": img_path, "points": pts, "width": w, "height": h})
        stats["images"] += 1
        stats["points"] += len(pts)
    return items, stats


def convert_pst_csv(images: list[Path], csv_path: Path, args) -> tuple[list, Counter]:
    """PST：imageName,X,Y,class —— 本身就是点标注，坐标原样使用。"""
    import csv as _csv

    pts_by_name: dict[str, list] = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            name = row.get("imageName") or row.get("image")
            if not name:
                continue
            pts_by_name.setdefault(Path(name).name, []).append(
                (float(row["X"]), float(row["Y"])))

    items, stats = [], Counter()
    for img_path in images:
        pts = pts_by_name.get(img_path.name, [])
        w, h = read_size(img_path)
        items.append({"src": img_path, "points": pts, "width": w, "height": h})
        stats["images"] += 1
        stats["points"] += len(pts)
    return items, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", required=True,
                    choices=["voc", "coco", "yolo", "pst-csv"])
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True, help="注释目录或单文件")
    ap.add_argument("--out", required=True, help="图片输出目录（会复制/软链）")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--copy", action="store_true", help="复制图片（默认建硬链接，省空间）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-points", type=int, default=0,
                    help="过滤点数少于该值的图（0 表示保留空样本作为负样本）")
    args = ap.parse_args()

    images = list_images(Path(args.images))
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"未找到图片：{args.images}")

    ann_path = Path(args.ann)
    if args.format == "voc":
        items, stats = convert_voc(images, ann_path, args)
    elif args.format == "coco":
        items, stats = convert_coco(images, ann_path, args)
    elif args.format == "yolo":
        items, stats = convert_yolo(images, ann_path, args)
    else:
        items, stats = convert_pst_csv(images, ann_path, args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for it in items:
        if args.min_points and len(it["points"]) < args.min_points:
            stats["filtered_empty"] += 1
            continue
        dst = out_dir / it["src"].name
        if not dst.exists():
            if args.copy:
                shutil.copy2(it["src"], dst)
            else:
                try:
                    dst.hardlink_to(it["src"])
                except (OSError, AttributeError):
                    shutil.copy2(it["src"], dst)
        records.append({
            "file_name": it["src"].name,
            "points": [[round(float(x), 2), round(float(y), 2)] for x, y in it["points"]],
            "width": it["width"], "height": it["height"],
        })

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"meta": {"beta": 0.3, "k": 3, "stride": 4,
                            "source": f"{args.format}:{ann_path.name}"},
                   "items": records}, f, ensure_ascii=False, indent=2)

    counts = [len(r["points"]) for r in records]
    print(f"图片 {len(records)} 张，点 {sum(counts)} 个，"
          f"平均 {sum(counts)/max(1,len(records)):.1f} 点/图，"
          f"最少 {min(counts) if counts else 0}，最多 {max(counts) if counts else 0}")
    print(f"统计：{dict(stats)}")
    print(f"图片 -> {out_dir}\n标注 -> {out_json}")
    print("\n提示：转换完成后用 tools/split_dataset.py 划分 train/val。")


if __name__ == "__main__":
    main()
