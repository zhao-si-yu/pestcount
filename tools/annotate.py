"""点标注工具（tkinter，零第三方 GUI 依赖）。

用法::

    python tools/annotate.py --images data/images/boards --out data/annotations/train.json

操作：
    左键        添加一个虫体中心点
    右键        删除最近的标注点（8 像素内）
    滚轮        以鼠标位置为中心缩放
    n / p       下一张 / 上一张（自动保存）
    u           撤销上一个点
    c           清空当前图所有点
    s           立即保存
    Ctrl+滚轮   微调

为什么自己写一个：
    标框类工具（labelme 等）给的是 bbox，而密度估计需要中心点；
    用 box 中心点代替会引入系统误差（虫体弯曲、遮挡时框中心偏移）。
    本工具直接输出点坐标，并严格沿用 dataset 的 JSON 格式，标完即可训练。
"""

from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pestcount.utils import imread_unicode

CANVAS_W, CANVAS_H = 1180, 760
DEL_RADIUS = 10          # 右键删除的命中半径（屏幕像素）


class Annotator:
    def __init__(self, root: tk.Tk, images: list[Path], out_json: Path, start: int = 0):
        self.root = root
        self.images = images
        self.out_json = out_json
        self.idx = max(0, min(start, len(images) - 1))

        self.records: dict[str, list[list[float]]] = {}
        self._load_existing()

        self.points: list[list[float]] = []
        self.zoom = 1.0
        self.offset = (0.0, 0.0)      # 图像左上角在画布中的位置
        self.scale = 1.0              # 综合缩放（fit * zoom）
        self.photo: ImageTk.PhotoImage | None = None

        self._build_ui()
        self.load_image()

    # ---------- 数据 ----------
    def _load_existing(self) -> None:
        if not self.out_json.exists():
            return
        try:
            with open(self.out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            for it in data.get("items", []):
                self.records[it["file_name"]] = it.get("points", [])
        except Exception as exc:                       # 坏文件不应阻塞标注
            messagebox.showwarning("读取标注失败", f"{exc}\n将从头开始标注")

    def _rel(self, path: Path) -> str:
        try:
            return path.relative_to(self.images).as_posix()
        except ValueError:
            return path.name

    def save(self, silent: bool = True) -> None:
        items = []
        for p in self.images:
            rel = self._rel(p)
            if rel in self.records:
                pts = self.records[rel]
            elif p == self.images[self.idx]:
                pts = self.points
            else:
                continue
            h = w = 0
            im = imread_unicode(p)
            if im is not None:
                h, w = im.shape[:2]
            items.append({"file_name": rel, "points": pts, "width": w, "height": h})
        payload = {"meta": {"beta": 0.3, "k": 3, "stride": 4}, "items": items}
        self.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        if not silent:
            self.status(f"已保存 {len(items)} 张 -> {self.out_json.name}")

    # ---------- UI ----------
    def _build_ui(self) -> None:
        self.root.title("PestCount 点标注工具")
        self.info = tk.Label(self.root, text="", font=("Microsoft YaHei", 11),
                             anchor="w", padx=8, pady=4)
        self.info.pack(fill="x")

        self.canvas = tk.Canvas(self.root, width=CANVAS_W, height=CANVAS_H,
                                bg="#1e1e1e", highlightthickness=0, cursor="crosshair")
        self.canvas.pack()

        bar = tk.Frame(self.root)
        bar.pack(fill="x", pady=4)
        for text, cmd in (("上一张 (p)", self.prev), ("下一张 (n)", self.next),
                          ("撤销 (u)", self.undo), ("清空 (c)", self.clear),
                          ("保存 (s)", lambda: self.save(silent=False)), ("退出", self.quit)):
            tk.Button(bar, text=text, command=cmd, width=12).pack(side="left", padx=4)

        self.canvas.bind("<Button-1>", self.on_left)
        self.canvas.bind("<Button-3>", self.on_right)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        for key, fn in (("n", self.next), ("p", self.prev), ("u", self.undo),
                        ("c", self.clear), ("s", lambda: self.save(silent=False))):
            self.root.bind(key, lambda e, f=fn: f())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def status(self, extra: str = "") -> None:
        self.info.config(
            text=f"[{self.idx+1}/{len(self.images)}] {self._rel(self.images[self.idx])}   "
                 f"点数 {len(self.points)}   zoom {self.zoom:.2f}x   {extra}")

    # ---------- 显示 ----------
    def load_image(self) -> None:
        path = self.images[self.idx]
        img = imread_unicode(path)
        if img is None:
            self.status("读取失败")
            return
        self.rgb = img[:, :, ::-1].copy()
        self.h, self.w = self.rgb.shape[:2]
        self.points = list(self.records.get(self._rel(path), []))
        self.zoom = 1.0
        self.redraw()
        self.status()

    def redraw(self) -> None:
        fit = min(CANVAS_W / self.w, CANVAS_H / self.h)
        self.scale = fit * self.zoom
        dw, dh = int(self.w * self.scale), int(self.h * self.scale)
        ox = (CANVAS_W - dw) / 2
        oy = (CANVAS_H - dh) / 2
        self.offset = (ox, oy)

        pil = Image.fromarray(self.rgb).resize((dw, dh), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)
        r = max(3, int(min(dw, dh) * 0.006))
        for x, y in self.points:
            sx, sy = ox + x * self.scale, oy + y * self.scale
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r,
                                    outline="#00ff88", width=2, tags="pt")
        self.status()

    # ---------- 交互 ----------
    def _to_image(self, ex: float, ey: float) -> tuple[float, float]:
        ox, oy = self.offset
        x = (ex - ox) / self.scale
        y = (ey - oy) / self.scale
        return float(np.clip(x, 0, self.w - 1)), float(np.clip(y, 0, self.h - 1))

    def on_left(self, event: tk.Event) -> None:
        x, y = self._to_image(event.x, event.y)
        self.points.append([round(x, 2), round(y, 2)])
        self.redraw()

    def on_right(self, event: tk.Event) -> None:
        if not self.points:
            return
        ox, oy = self.offset
        best, best_d = -1, DEL_RADIUS ** 2
        for i, (x, y) in enumerate(self.points):
            d = (ox + x * self.scale - event.x) ** 2 + (oy + y * self.scale - event.y) ** 2
            if d < best_d:
                best, best_d = i, d
        if best >= 0:
            self.points.pop(best)
            self.redraw()

    def on_wheel(self, event: tk.Event) -> None:
        step = 1.15 if event.delta > 0 else 1 / 1.15
        self.zoom = float(np.clip(self.zoom * step, 1.0, 8.0))
        self.redraw()

    def undo(self) -> None:
        if self.points:
            self.points.pop()
            self.redraw()

    def clear(self) -> None:
        if self.points and not messagebox.askyesno("清空", "清空当前图的全部点？"):
            return
        self.points = []
        self.redraw()

    def _commit(self) -> None:
        self.records[self._rel(self.images[self.idx])] = list(self.points)
        self.save(silent=True)

    def next(self) -> None:
        self._commit()
        if self.idx + 1 < len(self.images):
            self.idx += 1
            self.load_image()
        else:
            self.status("已经是最后一张")

    def prev(self) -> None:
        self._commit()
        if self.idx > 0:
            self.idx -= 1
            self.load_image()

    def quit(self) -> None:
        try:
            self._commit()
        finally:
            self.root.destroy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, default="data/images/boards")
    ap.add_argument("--out", type=str, default="data/annotations/train.json")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    folder = Path(args.images)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])
    if not images:
        raise SystemExit(f"目录下没有图片：{folder}")

    root = tk.Tk()
    Annotator(root, images, Path(args.out), start=args.start)
    root.mainloop()


if __name__ == "__main__":
    main()
