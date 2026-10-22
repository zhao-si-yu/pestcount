# 数据目录约定

```
data/
├── images/
│   ├── boards/            # 真实诱捕板照片（手机拍，原图即可）
│   └── synthetic/         # make_synthetic.py 生成的合成图
└── annotations/
    ├── all.json           # 人工标注汇总
    ├── synthetic.json     # 合成数据标注（脚本自动生成）
    ├── train.json
    ├── val.json
    └── test.json
```

## 为什么不把图片放进仓库

* 诱捕板原图动辄 3~8 MB/张，几百张就把仓库撑爆，clone 体验极差；
* 采样地点、作物、时间属于田间试验信息，公开需谨慎。

推荐做法：**图片放 Release 附件或网盘，仓库只放标注 JSON 和生成脚本**。
别人 clone 后按 README 放好 `data/images/` 即可复现。

## 标注 JSON 格式

```json
{
  "meta": {"beta": 0.3, "k": 3, "stride": 4},
  "items": [
    {
      "file_name": "boards/IMG_0001.jpg",
      "points": [[120.5, 88.0], [301.2, 402.7]],
      "width": 1920,
      "height": 1440
    }
  ]
}
```

* `file_name` 相对于 `--data-root`（默认 `data/images`）；
* `points` 是虫体**中心点** `(x, y)`，x 为列、y 为行，单位像素；
* `meta.stride = 4` 表示密度图在 1/4 分辨率上生成，需与模型输出步长一致。

## 标注规范（保证一致性）

1. 每只虫标**一个中心点**，不要画框、不要标尺寸；
2. 虫体相互重叠时**分别标注**，被遮挡的也要标（密度估计依赖数量而非可见面积）；
3. 图像边缘被截断超过 50% 的目标**不标**；
4. 无法确认是虫体的杂质（灰尘、霉斑）**不标**；
5. 完全干净的板是**有效样本**（负样本），请保留一部分，用于压制误检。
