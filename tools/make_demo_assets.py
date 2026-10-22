"""生成演示用素材（模拟黄板背景 + 模拟虫体贴图）。

用途：
    1. 让任何人 clone 后无需真实数据就能跑通「合成 -> 训练 -> 评估」全流程；
    2. 作为 CI / 冒烟测试的夹具。

注意：这里生成的是**程序化模拟图**，只用于验证代码管线是否跑通，
不能代表真实精度。真实指标必须在自己的诱捕板数据上跑。

用法::

    python tools/make_demo_assets.py --boards 3 --insects 24
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_board(w: int, h: int, rng: np.random.Generator, hue_shift: float = 0.0):
    """模拟黄色粘虫板：底色 + 方向性光照 + 污渍 + 颗粒噪点 + 渐晕。"""
    base_bgr = np.array([60 + hue_shift, 215, 235], dtype=np.float32)   # 黄
    img = np.tile(base_bgr, (h, w, 1))

    # 光照不均：叠加低频渐变
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = (xx / w) * rng.uniform(-35, 35) + (yy / h) * rng.uniform(-25, 25)
    img += grad[..., None]

    # 污渍/胶痕
    for _ in range(rng.integers(6, 16)):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = rng.integers(10, max(12, min(w, h) // 6))
        color = base_bgr + rng.uniform(-45, 25, size=3)
        overlay = img.copy()
        cv2.circle(overlay, (int(cx), int(cy)), int(r), color.tolist(), -1)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    # 颗粒噪声（高 ISO / 压缩）
    img += rng.normal(0, 5.5, img.shape)

    # 渐晕
    mask = 1.0 - 0.35 * (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2))
    img *= mask[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def make_insect(size: int, rng: np.random.Generator):
    """模拟虫体：椭圆身体 + 头胸 + 触角，输出带 alpha 的 RGBA。"""
    s = size
    canvas = np.zeros((s, s, 4), dtype=np.uint8)
    cx, cy = s // 2, s // 2

    body_len = int(s * rng.uniform(0.62, 0.86))
    body_w = int(s * rng.uniform(0.30, 0.46))
    angle = rng.uniform(0, 180)
    color = np.array([rng.integers(15, 55), rng.integers(20, 70), rng.integers(25, 90), 255],
                     dtype=np.uint8)

    cv2.ellipse(canvas, (cx, cy + s // 10), (body_w // 2, body_len // 2),
                angle, 0, 360, color.tolist(), -1, cv2.LINE_AA)
    # 头胸
    head_c = np.array([rng.integers(10, 40), rng.integers(15, 45), rng.integers(20, 55), 255],
                      dtype=np.uint8)
    cv2.circle(canvas, (cx, cy - body_len // 3), max(2, body_w // 3),
               head_c.tolist(), -1, cv2.LINE_AA)
    # 触角 / 足
    for _ in range(rng.integers(2, 5)):
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(s * 0.25, s * 0.48)
        cv2.line(canvas, (cx, cy),
                 (int(cx + r * np.cos(a)), int(cy + r * np.sin(a))),
                 head_c.tolist(), 1, cv2.LINE_AA)

    # 轻微模糊，模拟景深
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.6)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=3)
    ap.add_argument("--insects", type=int, default=24)
    ap.add_argument("--board-dir", type=str, default="assets/boards")
    ap.add_argument("--insect-dir", type=str, default="assets/insects")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    bdir, idir = Path(args.board_dir), Path(args.insect_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    idir.mkdir(parents=True, exist_ok=True)

    for i in range(args.boards):
        w, h = int(rng.integers(900, 1300)), int(rng.integers(700, 1000))
        img = make_board(w, h, rng, hue_shift=float(rng.uniform(-25, 25)))
        # 用 imencode 写文件，避免 Windows 下中文路径问题
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            buf.tofile(str(bdir / f"demo_board_{i:02d}.jpg"))
    print(f"生成背景板 {args.boards} 张 -> {bdir}")

    for i in range(args.insects):
        s = int(rng.integers(18, 44))
        rgba = make_insect(s, rng)
        ok, buf = cv2.imencode(".png", rgba)
        if ok:
            buf.tofile(str(idir / f"demo_insect_{i:02d}.png"))
    print(f"生成虫体贴图 {args.insects} 张 -> {idir}")


if __name__ == "__main__":
    main()
