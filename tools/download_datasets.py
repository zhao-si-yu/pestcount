"""公开数据集下载器（带断点续传与格式提示）。

用法::

    python tools/download_datasets.py --list
    python tools/download_datasets.py --get pst      # 断点续传，可多次执行
    python tools/download_datasets.py --get yst
    python tools/download_datasets.py --extract pst

注意：这些站点在国内网络下可能很慢或不可达（Zenodo / HuggingFace 常见）。
`--get` 支持 Ctrl+C 中断后重新执行，会从断点继续，不必重下。
建议挂到夜间执行，或使用学校网络/代理。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

RAW = Path("data/raw")

DATASETS = {
    "pst": {
        "name": "PST - Pest Sticky Traps（白粉虱黄板，点标注）",
        "url": "https://zenodo.org/records/7801239/files/pest-sticky-traps.zip?download=1",
        "file": "pest-sticky-traps.zip",
        "size": "144 MB",
        "ann": "点标注 CSV（imageName, X, Y, class）",
        "images": "黄板照片，官方已划分 train/test",
        "license": "CC BY 4.0",
        "fit": "★ 最匹配：原生点标注，可直接训练，无需转换",
        "convert": ("--format pst-csv --images data/raw/pest-sticky-traps/train "
                    "--ann data/raw/pest-sticky-traps/train/annotations.csv "
                    "--out data/images/pst --out-json data/annotations/pst.json"),
    },
    "yst": {
        "name": "Yellow Sticky Traps（黄板 3 类，改进版框标注）",
        "url": ("https://github.com/GEMIT-Institute-of-HS-Niederrhein/"
                "yellow-sticky-traps-dataset/archive/refs/heads/main.zip"),
        "file": "yellow-sticky-traps.zip",
        "size": "约 1.5 GB（284 张 5184x3456 原图）",
        "ann": "PASCAL VOC XML 框标注，8114 个框",
        "images": "284 张黄板，3 类（白粉虱 5807 / Macrolophus 1619 / Nesidiocoris 688）",
        "license": "见仓库 LICENSE",
        "fit": "★ 场景最贴近诱捕板，但需框转点",
        "convert": ("--format voc "
                    "--images data/raw/yellow-sticky-traps-dataset-main/images "
                    "--ann data/raw/yellow-sticky-traps-dataset-main/annotations "
                    "--out data/images/boards --out-json data/annotations/yst.json"),
    },
    "kaggle-yst": {
        "name": "Yellow Sticky Traps（Kaggle 原始版，CC0）",
        "url": "kaggle datasets download -d friso1987/yellow-sticky-traps",
        "file": "yellow-sticky-traps.zip",
        "size": "808 MB",
        "ann": "PASCAL VOC XML 框标注（5591 白粉虱 + 1312 + 510）",
        "images": "284 张，565 个文件",
        "license": "CC0 公共领域",
        "fit": "许可最宽松，适合公开发布实验；需 kaggle CLI 与 API token",
        "convert": ("--format voc --images <解压目录>/stickytraps "
                    "--ann <解压目录>/stickytraps "
                    "--out data/images/boards --out-json data/annotations/kaggle_yst.json"),
    },
    "agripest": {
        "name": "AgriPest（田间害虫 14 类，含计数子任务）",
        "url": "https://github.com/liuliu66/AgriPest （当前返回 404，需按论文找新地址）",
        "file": "-",
        "size": "49.7K 图 / 264.7K 框",
        "ann": "框标注，官方定义了 pest counting 子任务与四种场景子集",
        "images": "小麦/水稻/玉米/油菜田间的微小害虫",
        "license": "CC BY 4.0（论文声明）",
        "fit": "「农田害虫」语义最匹配，但仓库已失效，需自行检索",
        "convert": "--format voc/coco --images ... --ann ... --out ... --out-json ...",
    },
    "ip102": {
        "name": "IP102（102 类害虫分类）",
        "url": "https://github.com/xpwu95/IP102",
        "file": "-",
        "size": "75,222 图，其中 19,000 带框",
        "ann": "分类标签 + 部分框标注",
        "images": "单虫特写，非板面场景",
        "license": "学术使用",
        "fit": "不适合计数（无场景、无密度分布），可作为虫种识别分支",
        "convert": "不适用",
    },
}


def cmd_list() -> None:
    print("公开害虫计数/检测数据集：\n")
    for key, d in DATASETS.items():
        print(f"[{key}] {d['name']}")
        print(f"    规模    : {d['size']}")
        print(f"    标注    : {d['ann']}")
        print(f"    图像    : {d['images']}")
        print(f"    许可    : {d['license']}")
        print(f"    适配    : {d['fit']}")
        print(f"    地址    : {d['url']}\n")


def cmd_get(key: str) -> None:
    d = DATASETS[key]
    RAW.mkdir(parents=True, exist_ok=True)
    dst = RAW / d["file"]

    if d["url"].startswith("kaggle"):
        print("该数据集需 Kaggle 账号：\n"
              "  1) pip install kaggle\n"
              "  2) 在 kaggle.com/account 生成 API Token，放到 ~/.kaggle/kaggle.json\n"
              f"  3) {d['url']}\n"
              f"目标目录：{RAW}")
        return

    print(f"下载 {d['name']}\n -> {dst}\n（可中断，重新执行会断点续传）\n")
    cmd = ["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "3",
           "-o", str(dst), d["url"]]
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("未找到 curl，请用浏览器直接下载：", d["url"])
        return

    if dst.exists():
        mb = dst.stat().st_size / 1024 ** 2
        print(f"\n完成：{dst}  {mb:.1f} MB")
        print(f"下一步：python tools/download_datasets.py --extract {key}")


def cmd_extract(key: str) -> None:
    d = DATASETS[key]
    src = RAW / d["file"]
    if not src.exists():
        raise SystemExit(f"压缩包不存在：{src}（先执行 --get {key}）")
    print(f"解压 {src} ...")
    with zipfile.ZipFile(src) as z:
        z.extractall(RAW)
    print(f"已解压到 {RAW}")
    print("\n转换标注为点格式：\n  python tools/convert_annotations.py "
          + d["convert"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--get", choices=list(DATASETS))
    ap.add_argument("--extract", choices=list(DATASETS))
    args = ap.parse_args()

    if args.list or (not args.get and not args.extract):
        cmd_list()
    if args.get:
        cmd_get(args.get)
    if args.extract:
        cmd_extract(args.extract)


if __name__ == "__main__":
    main()
