"""合成数据生成：把单只害虫贴到真实诱捕板背景上，自动产出点标注。

动机
----
点监督计数的主要瓶颈是标注量。但诱捕板场景有个天然优势：
虫体是「可分离的小目标」，把单只虫抠图后按随机姿态/尺度/光照
贴回不同的板背景，就能批量生成带精确点标注的样本。

这类合成样本的域分布和真实照片仍有差距（光照方向、景深、粘连），
所以推荐两阶段用法：
    1) 合成集预训练，让网络学会「虫体的外观先验」；
    2) 少量真实标注微调，把域差距补上。

用法::

    python tools/make_synthetic.py \
        --boards assets/boards --insects assets/insects \
        --out-images data/images/synthetic --out-json data/annotations/synthetic.json \
        --num 600 --per-image 12-40

贴图要求：优先使用带 alpha 通道的 PNG；若为普通图片，默认用
Otsu 阈值把「比板面暗」的像素当作虫体前景（黄板粘虫的典型成像）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pestcount.utils import imread_unicode, imwrite_unicode


def list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


def load_sticker(path: Path, auto_mask: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
    """读取单虫贴图，返回 (BGR, alpha[0..1])。"""
    if path.suffix.lower() == ".png":
        raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        if raw.ndim == 3 and raw.shape[2] == 4:
            bgr = raw[:, :, :3].astype(np.float32)
            alpha = raw[:, :, 3].astype(np.float32) / 255.0
            return bgr, alpha

    img = imread_unicode(path)
    if img is None:
        return None
    if not auto_mask:
        return img.astype(np.float32), np.ones(img.shape[:2], dtype=np.float32)

    # 自动抠图：虫体通常比板面暗，用 Otsu 分离后取暗侧前景
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    alpha = cv2.GaussianBlur(th.astype(np.float32) / 255.0, (3, 3), 0)
    if alpha.mean() > 0.75:            # 前景占比过高说明抠反了，丢弃该贴图
        return None
    # 收紧到前景外接框，避免带入大片背景
    ys, xs = np.where(th > 0)
    if xs.size < 30:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1].astype(np.float32), alpha[y0:y1, x0:x1]


def random_transform(sticker: np.ndarray, alpha: np.ndarray,
                     target_px: float, rng: random.Random):
    """随机缩放/旋转/调色/模糊，模拟不同拍摄条件。"""
    h, w = sticker.shape[:2]
    scale = target_px / max(h, w)
    nw, nh = max(3, int(w * scale)), max(3, int(h * scale))
    st = cv2.resize(sticker, (nw, nh), interpolation=cv2.INTER_AREA)
    al = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)

    angle = rng.uniform(0, 360)
    diag = int(np.ceil(np.hypot(nw, nh)))
    M = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1.0)
    M[0, 2] += (diag - nw) / 2
    M[1, 2] += (diag - nh) / 2
    st = cv2.warpAffine(st, M, (diag, diag), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)
    al = cv2.warpAffine(al, M, (diag, diag), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 亮度/对比度/色偏：不同时段的光照
    st = st * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)
    st = st + np.array([rng.uniform(-10, 10) for _ in range(3)], dtype=np.float32)
    if rng.random() < 0.25:
        st = cv2.GaussianBlur(st, (3, 3), 0)
    st = np.clip(st, 0, 255)
    return st, np.clip(al, 0, 1)


def paste(board: np.ndarray, st: np.ndarray, al: np.ndarray,
          cx: int, cy: int) -> None:
    """按 alpha 混合把贴图叠到背景上，中心对齐到 (cx, cy)。"""
    bh, bw = board.shape[:2]
    sh, sw = st.shape[:2]
    x0, y0 = cx - sw // 2, cy - sh // 2
    x1, y1 = x0 + sw, y0 + sh
    bx0, by0 = max(0, x0), max(0, y0)
    bx1, by1 = min(bw, x1), min(bh, y1)
    if bx0 >= bx1 or by0 >= by1:
        return
    sx0, sy0 = bx0 - x0, by0 - y0
    sx1, sy1 = sx0 + (bx1 - bx0), sy0 + (by1 - by0)
    roi = board[by0:by1, bx0:bx1].astype(np.float32)
    a = al[sy0:sy1, sx0:sx1][..., None]
    board[by0:by1, bx0:bx1] = (st[sy0:sy1, sx0:sx1] * a + roi * (1 - a)).astype(np.uint8)


def parse_range(text: str) -> tuple[int, int]:
    if "-" in text:
        a, b = text.split("-", 1)
        return int(a), int(b)
    v = int(text)
    return v, v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=str, default="assets/boards")
    ap.add_argument("--insects", type=str, default="assets/insects")
    ap.add_argument("--out-images", type=str, default="data/images/synthetic")
    ap.add_argument("--out-json", type=str, default="data/annotations/synthetic.json")
    ap.add_argument("--num", type=int, default=600, help="生成图片数量")
    ap.add_argument("--per-image", type=str, default="12-40", help="每张图的虫数范围")
    ap.add_argument("--size-range", type=str, default="0.010-0.035",
                    help="虫体短边占图像短边的比例范围")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    boards = list_images(Path(args.boards))
    insect_files = list_images(Path(args.insects))
    if not boards:
        raise SystemExit(f"未找到板背景图：{args.boards}（放几张干净黄板照片即可）")
    if not insect_files:
        raise SystemExit(f"未找到单虫贴图：{args.insects}（建议透明底 PNG）")

    stickers = [s for s in (load_sticker(p) for p in insect_files) if s is not None]
    if not stickers:
        raise SystemExit("所有贴图抠图失败，请改为提供带 alpha 通道的 PNG")

    out_img_dir = Path(args.out_images)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = parse_range(args.per_image)
    s_lo, s_hi = (float(v) for v in args.size_range.split("-"))

    items = []
    for i in range(args.num):
        board_path = boards[i % len(boards)]
        board = imread_unicode(board_path)
        if board is None:
            continue
        # 随机裁剪出不同视野，增加尺度多样性
        bh, bw = board.shape[:2]
        if rng.random() < 0.7:
            cw = rng.randint(int(bw * 0.55), bw)
            ch = rng.randint(int(bh * 0.55), bh)
            x0 = rng.randint(0, bw - cw)
            y0 = rng.randint(0, bh - ch)
            board = board[y0:y0 + ch, x0:x0 + cw].copy()
        else:
            board = board.copy()

        bh, bw = board.shape[:2]
        n = rng.randint(lo, hi)
        short = min(bh, bw)
        points = []
        for _ in range(n):
            st, al = random_transform(*rng.choice(stickers),
                                      target_px=short * rng.uniform(s_lo, s_hi),
                                      rng=rng)
            cx = rng.randint(0, bw - 1)
            cy = rng.randint(0, bh - 1)
            paste(board, st, al, cx, cy)
            points.append([float(cx), float(cy)])

        name = f"syn_{i:05d}.jpg"
        imwrite_unicode(out_img_dir / name, board, quality=args.jpeg_quality)
        items.append({
            "file_name": name,          # 相对 --out-images 的路径
            "points": points,
            "width": bw, "height": bh,
        })
        if (i + 1) % 50 == 0:
            print(f"  generated {i+1}/{args.num}", flush=True)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"meta": {"source": "synthetic", "beta": 0.3, "k": 3, "stride": 4},
                   "items": items}, f, ensure_ascii=False, indent=2)

    total = sum(len(it["points"]) for it in items)
    print(f"\n完成：{len(items)} 张，共 {total} 个虫体点标注")
    print(f"图像 -> {out_img_dir}")
    print(f"标注 -> {out_json}")


if __name__ == "__main__":
    main()
