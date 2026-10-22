# 端侧部署基准

- 测试设备：Intel64 Family 6 Model 186 Stepping 2, GenuineIntel / NVIDIA GeForce RTX 4060 Laptop GPU
- 输入尺寸：1024×1024
- 统计方式：warmup 10 次后连续 50 次，取均值与 P90（含前后处理外的纯网络前向）

| 版本 | 执行后端 | 参数量 | 体积(MB) | 延迟均值(ms) | 延迟P90(ms) | 显存(MB) |
|---|---|---|---|---|---|---|
| PyTorch FP32 | CUDA | 0.329M | 1.26 | 7.7 | 8.2 | 182 |
| ONNX FP32 | CPU | 0.329M | 1.27 | 93.5 | 97.7 | 0 |
| ONNX INT8 | CPU | 0.329M | 0.36 | 96.0 | 113.9 | 0 |
