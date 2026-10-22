"""导出 ONNX，并可选做 INT8 静态量化。

用法::

    # 仅导出 FP32
    python export_onnx.py --ckpt checkpoints/best.pt --out checkpoints/pestcount.onnx

    # 导出 + INT8 量化（需要 onnxruntime + 校准图片）
    python export_onnx.py --ckpt checkpoints/best.pt --int8 \
        --calib-json data/annotations/val.json --data-root data/images

依赖（未预装时先安装）::

    pip install onnx onnxruntime
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pestcount.model import build_model
from pestcount.predictor import load_checkpoint
from pestcount.utils import to_input_tensor


def build(ckpt: str, infer_size: int):
    blob = load_checkpoint(ckpt, "cpu")
    state = blob.get("model", blob)
    cfg = blob.get("args", {}) if isinstance(blob, dict) else {}
    width = float(cfg.get("width", 1.5))
    model = build_model(width=width).eval()
    model.load_state_dict(state)
    dummy = torch.zeros(1, 3, infer_size, infer_size)
    return model, dummy, width


def export_fp32(model, dummy, out_path: Path, dynamic: bool) -> None:
    import onnx  # noqa: F401  (显式依赖检查)

    axes = {"input": {2: "h", 3: "w"}, "density": {2: "h", 3: "w"}} if dynamic else None
    # dynamo=False：用 legacy exporter，权重内联进 .onnx 单文件。
    # torch>=2.7 的新导出器默认把权重写到外部 .onnx.data，部署时容易漏文件
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["density"],
        opset_version=17, do_constant_folding=True, dynamic_axes=axes,
        dynamo=False,
    )
    # 兜底：若仍产生外部数据文件，合并回单文件
    data_file = out_path.with_name(out_path.name + ".data")
    if data_file.exists():
        m = onnx.load(str(out_path), load_external_data=True)
        for tensor in m.graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                tensor.data_location = onnx.TensorProto.DEFAULT
                tensor.ClearField("external_data")
        onnx.save(m, str(out_path))
        data_file.unlink()
    print(f"[ok] ONNX 导出 {out_path}  {out_path.stat().st_size/1e6:.2f} MB  "
          f"dynamic_axes={'on' if dynamic else 'off'}")


def quantize_int8(fp32_path: Path, out_path: Path, calib_json: str,
                  data_root: str, infer_size: int, n_calib: int = 48) -> None:
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # 量化前先做官方预处理（shape inference + 常量折叠），
    # 否则 SiLU 的 Mul 节点在量化时会报 ShapeInferenceError
    pre_path = fp32_path.with_name(fp32_path.stem + "_pre.onnx")
    quant_pre_process(str(fp32_path), str(pre_path), skip_optimization=False,
                      skip_shape_inference=False)

    from pestcount.dataset import PointCountDataset, load_index
    from pestcount.utils import denormalize

    items = load_index(calib_json)["items"][:n_calib]
    ds = PointCountDataset(items, data_root, mode="val", infer_size=infer_size)

    def _square(t: "torch.Tensor") -> "np.ndarray":
        """固定形状导出的模型要求方形输入；校准图统一 resize 到方形。"""
        import cv2
        arr = t.numpy().transpose(1, 2, 0)
        arr = cv2.resize(arr, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
        return arr.transpose(2, 0, 1)[None].astype(np.float32)

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.data = [{"input": _square(ds[i]["image"])} for i in range(len(ds))]
            self.i = 0

        def get_next(self):
            if self.i >= len(self.data):
                return None
            item = self.data[self.i]
            self.i += 1
            return item

    quantize_static(str(pre_path), str(out_path), Reader(),
                    quant_format=QuantFormat.QDQ,      # x64 CPU 上 QDQ 有更好的算子融合
                    op_types_to_quantize=["Conv"],
                    per_channel=True, reduce_range=False)
    print(f"[ok] INT8 量化 {out_path}  {out_path.stat().st_size/1e6:.2f} MB")


def verify(fp32_path: Path, int8_path: Path | None, infer_size: int) -> None:
    """用同一张随机图对比 torch / onnx / int8 的输出，检查计数偏差。"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[skip] 未安装 onnxruntime，跳过一致性校验")
        return

    rng = np.random.default_rng(0)
    x = rng.random((1, 3, infer_size, infer_size)).astype(np.float32)

    def run(path: Path) -> float:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return float(sess.run(None, {"input": x})[0].sum())

    ref = run(fp32_path)
    print(f"onnx  fp32  计数={ref:.4f}")
    if int8_path and int8_path.exists():
        q = run(int8_path)
        print(f"onnx  int8  计数={q:.4f}  相对偏差={abs(q-ref)/max(ref,1e-6)*100:.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--out", type=str, default="checkpoints/pestcount.onnx")
    ap.add_argument("--infer-size", type=int, default=1024)
    ap.add_argument("--dynamic", action="store_true", help="动态输入尺寸（不利于端侧优化）")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib-json", type=str, default="data/annotations/val.json")
    ap.add_argument("--data-root", type=str, default="data/images")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, dummy, width = build(args.ckpt, args.infer_size)
    export_fp32(model, dummy, out_path, args.dynamic)

    int8_path = None
    if args.int8:
        int8_path = out_path.with_name(out_path.stem + "_int8.onnx")
        quantize_int8(out_path, int8_path, args.calib_json, args.data_root,
                      args.infer_size)

    verify(out_path, int8_path, args.infer_size)


if __name__ == "__main__":
    main()
