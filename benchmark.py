"""端侧部署基准测试：一次跑出可写进 README / 简历的实测表。

用法::

    python benchmark.py --ckpt checkpoints/best.pt --size 1024

输出（Markdown）：参数量、权重体积、PyTorch FP32 / ONNX FP32 / ONNX INT8
的单帧延迟、P50/P90、显存占用。

注意：
  * 延迟数字必须来自本机实测，不要手填；
  * 首次调用包含 CUDA 上下文与 cuDNN 自动调优开销，因此先 warmup 再统计；
  * 端侧（如树莓派/RK3588）请在目标设备上重跑本脚本，不要用桌面数据代替。
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from pestcount.model import build_model
from pestcount.predictor import load_checkpoint


def bench_torch(model, size: int, device: str, warmup: int = 10, iters: int = 50):
    x = torch.randn(1, 3, size, size, device=device)
    model.eval().to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    mem = torch.cuda.max_memory_allocated() / 1024 ** 2 if device == "cuda" else 0.0
    return times, mem


def bench_onnx(path: Path, size: int, warmup: int = 10, iters: int = 50):
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if "CUDAExecutionProvider" in ort.get_available_providers() \
        else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), providers=providers)
    x = np.random.rand(1, 3, size, size).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"input": x})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    return times, providers[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--out", type=str, default="docs/benchmark.md")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    blob = load_checkpoint(args.ckpt, "cpu")
    cfg = blob.get("args", {}) if isinstance(blob, dict) else {}
    width = float(cfg.get("width", 1.5))
    model = build_model(width=width).eval()
    model.load_state_dict(blob.get("model", blob))

    n_param = sum(p.numel() for p in model.parameters())
    torch_mb = sum(p.numel() * p.element_size() for p in model.state_dict().values()) / 1024 ** 2

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    times, mem = bench_torch(model, args.size, device)
    rows = [("PyTorch FP32", "CUDA" if device == "cuda" else "CPU", n_param, torch_mb,
             statistics.mean(times), np.percentile(times, 90), mem)]

    onnx_path = Path(args.ckpt).with_name("pestcount.onnx")
    int8_path = Path(args.ckpt).with_name("pestcount_int8.onnx")
    for path, label in ((onnx_path, "ONNX FP32"), (int8_path, "ONNX INT8")):
        if not path.exists():
            continue
        try:
            t, prov = bench_onnx(path, args.size)
            rows.append((label, prov.split("ExecutionProvider")[0], n_param,
                         path.stat().st_size / 1024 ** 2,
                         statistics.mean(t), np.percentile(t, 90), 0.0))
        except ImportError:
            print("[skip] 未安装 onnxruntime")

    lines = [
        f"# 端侧部署基准",
        "",
        f"- 测试设备：{platform.processor() or platform.machine()} / "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"- 输入尺寸：{args.size}×{args.size}",
        f"- 统计方式：warmup 10 次后连续 50 次，取均值与 P90（含前后处理外的纯网络前向）",
        "",
        "| 版本 | 执行后端 | 参数量 | 体积(MB) | 延迟均值(ms) | 延迟P90(ms) | 显存(MB) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, backend, p, mb, mean, p90, mem in rows:
        lines.append(f"| {name} | {backend} | {p/1e6:.3f}M | {mb:.2f} | "
                     f"{mean:.1f} | {p90:.1f} | {mem:.0f} |")

    text = "\n".join(lines)
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
