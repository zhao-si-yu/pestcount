"""可视化演示程序（tkinter，零第三方 GUI 依赖）。

用法::

    python app.py                      # 自动读取 checkpoints/best.pt
    python app.py --ckpt path/to.pt

功能：
    * 单张：选图 -> 显示原图 / 预测热力图 / 计数
    * 批量：选文件夹 -> 逐张推理 -> 导出 CSV（文件, 预测计数, 耗时）
    * 权重与推理尺寸可在界面上切换
"""

from __future__ import annotations

import argparse
import csv
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from pestcount.predictor import Predictor
from pestcount.utils import heatmap_overlay, imread_unicode

PREVIEW_W, PREVIEW_H = 520, 420
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class App:
    def __init__(self, root: tk.Tk, ckpt: str | None):
        self.root = root
        self.root.title("农田害虫智能计数系统 · Point-supervised Density Estimation")
        self.root.geometry("1120x760")
        self.predictor: Predictor | None = None
        self.current: np.ndarray | None = None
        self.current_path: Path | None = None
        self.last_result = None

        self._build()
        if ckpt:
            self.ckpt_var.set(ckpt)
            self.load_model(silent=True)

    # ---------- UI ----------
    def _build(self) -> None:
        top = tk.Frame(self.root, padx=10, pady=8)
        top.pack(fill="x")

        tk.Label(top, text="权重").pack(side="left")
        self.ckpt_var = tk.StringVar(value="checkpoints/best.pt")
        tk.Entry(top, textvariable=self.ckpt_var, width=38).pack(side="left", padx=4)
        tk.Button(top, text="浏览", command=self.pick_ckpt).pack(side="left")
        tk.Button(top, text="加载模型", command=self.load_model).pack(side="left", padx=4)

        tk.Label(top, text="推理尺寸").pack(side="left", padx=(16, 2))
        self.size_var = tk.StringVar(value="1024")
        ttk.Combobox(top, textvariable=self.size_var, width=6,
                     values=["512", "768", "1024", "1280", "1536"]).pack(side="left")

        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=10)

        self.left = tk.Canvas(body, width=PREVIEW_W, height=PREVIEW_H,
                              bg="#f2f2f2", highlightthickness=1,
                              highlightbackground="#cccccc")
        self.left.grid(row=0, column=0, padx=6, pady=6)
        self.right = tk.Canvas(body, width=PREVIEW_W, height=PREVIEW_H,
                               bg="#f2f2f2", highlightthickness=1,
                               highlightbackground="#cccccc")
        self.right.grid(row=0, column=1, padx=6, pady=6)

        self.count_label = tk.Label(body, text="计数 —",
                                    font=("Microsoft YaHei", 26, "bold"), fg="#c0392b")
        self.count_label.grid(row=1, column=0, columnspan=2, pady=(2, 8))

        bar = tk.Frame(self.root, padx=10, pady=6)
        bar.pack(fill="x")
        tk.Button(bar, text="选择图片", width=14, command=self.pick_image).pack(side="left")
        tk.Button(bar, text="重新推理", width=12, command=self.run_single).pack(side="left", padx=6)
        tk.Button(bar, text="批量处理文件夹", width=18, command=self.run_batch).pack(side="left", padx=6)
        tk.Button(bar, text="导出当前结果", width=14, command=self.export_current).pack(side="left", padx=6)

        self.status = tk.Label(self.root, text="就绪", anchor="w", padx=10,
                               fg="#555555", font=("Microsoft YaHei", 9))
        self.status.pack(fill="x", pady=(0, 6))

    def set_status(self, text: str) -> None:
        self.status.config(text=text)
        self.root.update_idletasks()

    # ---------- 模型 ----------
    def pick_ckpt(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pt *.pth"), ("All", "*.*")])
        if path:
            self.ckpt_var.set(path)
            self.load_model()

    def load_model(self, silent: bool = False) -> None:
        ckpt = self.ckpt_var.get().strip()
        if not Path(ckpt).exists():
            if not silent:
                messagebox.showerror("找不到权重", f"{ckpt}\n\n请先运行 train.py 训练，或手动选择 .pt 文件")
            self.set_status("未加载权重（可先训练模型）")
            return
        try:
            self.predictor = Predictor(ckpt, device="cuda",
                                       infer_size=int(self.size_var.get()))
            n = sum(p.numel() for p in self.predictor.model.parameters())
            self.set_status(f"已加载 {Path(ckpt).name}  参数 {n/1e6:.3f}M  设备 {self.predictor.device}")
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc))

    # ---------- 推理 ----------
    def pick_image(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Images", " ".join(f"*{e}" for e in EXTS)), ("All", "*.*")])
        if not path:
            return
        img = imread_unicode(path)
        if img is None:
            messagebox.showerror("读取失败", path)
            return
        self.current = img[:, :, ::-1].copy()
        self.current_path = Path(path)
        self.run_single()

    def _show(self, canvas: tk.Canvas, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        s = min(PREVIEW_W / w, PREVIEW_H / h)
        pil = Image.fromarray(rgb).resize((max(1, int(w * s)), max(1, int(h * s))),
                                          Image.BILINEAR)
        canvas.delete("all")
        canvas.image = ImageTk.PhotoImage(pil)
        canvas.create_image(PREVIEW_W / 2, PREVIEW_H / 2, anchor="center",
                            image=canvas.image)

    def run_single(self) -> None:
        if self.predictor is None:
            self.load_model()
        if self.predictor is None or self.current is None:
            return
        self.predictor.infer_size = int(self.size_var.get())
        t0 = time.perf_counter()
        out = self.predictor.predict(self.current)
        dt = (time.perf_counter() - t0) * 1000
        self.last_result = {"path": self.current_path, "count": out["count"],
                            "ms": dt, "size": out["size"]}

        vis = heatmap_overlay(self.current, out["density"])
        self._show(self.left, self.current)
        self._show(self.right, vis[:, :, ::-1])
        self.count_label.config(text=f"计数 {out['count']:.1f}")
        self.set_status(f"{self.current_path.name}  推理 {dt:.1f} ms  "
                        f"输入 {out['size']}  密度图积分 {out['count']:.2f}")

    def run_batch(self) -> None:
        if self.predictor is None:
            self.load_model()
        if self.predictor is None:
            return
        folder = filedialog.askdirectory(title="选择待计数图片所在文件夹")
        if not folder:
            return
        files = sorted([p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS])
        if not files:
            messagebox.showinfo("提示", "该文件夹下没有图片")
            return
        self.predictor.infer_size = int(self.size_var.get())

        rows, counts = [], []
        for i, path in enumerate(files):
            self.set_status(f"批量处理中 {i+1}/{len(files)} ...")
            img = imread_unicode(path)
            if img is None:
                continue
            t0 = time.perf_counter()
            out = self.predictor.predict(img[:, :, ::-1])
            rows.append([path.name, f"{out['count']:.2f}",
                         f"{(time.perf_counter()-t0)*1000:.1f}"])
            counts.append(out["count"])

        out_csv = Path(folder) / "pest_count_result.csv"
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "predicted_count", "latency_ms"])
            w.writerows(rows)
            w.writerow(["TOTAL", f"{sum(counts):.2f}", ""])
        self.set_status(f"批量完成 {len(rows)} 张，合计 {sum(counts):.0f} 只 -> {out_csv.name}")
        messagebox.showinfo("完成", f"共处理 {len(rows)} 张\n合计 {sum(counts):.0f} 只\n\nCSV：{out_csv}")

    def export_current(self) -> None:
        if self.last_result is None:
            messagebox.showinfo("提示", "请先推理一张图片")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            initialfile="result.csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "predicted_count", "latency_ms", "input_size"])
            w.writerow([self.last_result["path"].name,
                        f"{self.last_result['count']:.2f}",
                        f"{self.last_result['ms']:.1f}",
                        f"{self.last_result['size']}"])
        self.set_status(f"已导出 {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    args = ap.parse_args()
    root = tk.Tk()
    App(root, args.ckpt)
    root.mainloop()


if __name__ == "__main__":
    main()
