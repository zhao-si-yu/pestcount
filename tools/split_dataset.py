"""把一份完整标注按比例划分为 train / val（可选 test）。

用法::

    python tools/split_dataset.py --src data/annotations/all.json \
        --out-dir data/annotations --ratios 0.8 0.15 0.05 --seed 42

划分按「每张图的虫数」分层：先按计数排序，再轮流分配到各 split，
避免出现「训练集全是稀疏板、验证集全是密集板」这种分布偏移——
这种偏移在计数任务里会把 MAE 虚高好几个点，而且很难从日志看出来。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="完整标注 json")
    ap.add_argument("--out-dir", type=str, default="data/annotations")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.15, 0.05],
                    help="train val test 比例，三者之和为 1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-test", action="store_true", help="只分 train/val")
    args = ap.parse_args()

    with open(args.src, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    meta = data.get("meta", {"beta": 0.3, "k": 3, "stride": 4})
    if not items:
        raise SystemExit("标注文件里没有 items")

    rng = random.Random(args.seed)
    # 按计数分层：排序后打散次序再轮流分配，保证各 split 的密度分布接近
    order = sorted(range(len(items)), key=lambda i: len(items[i].get("points", [])))
    order = [order[i] for i in rng.sample(range(len(order)), len(order))]

    n = len(order)
    ratios = args.ratios[:2] + ([0.0] if args.no_test else [args.ratios[2]])
    total = sum(ratios)
    ratios = [r / total for r in ratios]
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))

    splits = {
        "train": [items[i] for i in order[:n_train]],
        "val": [items[i] for i in order[n_train:n_train + n_val]],
        "test": [items[i] for i in order[n_train + n_val:]],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        if name == "test" and not subset:
            continue
        path = out_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "items": subset}, f, ensure_ascii=False, indent=2)
        cnt = [len(it.get("points", [])) for it in subset]
        print(f"{name:5s} {len(subset):4d} 张  "
              f"平均虫数 {sum(cnt)/max(1,len(cnt)):.1f}  -> {path}")


if __name__ == "__main__":
    main()
