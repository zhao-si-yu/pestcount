"""训练入口。

用法::

    python train.py --data-root data/images \
                    --train-json data/annotations/train.json \
                    --val-json data/annotations/val.json \
                    --epochs 120 --batch 8 --crop 512

说明：
  * 训练日志一律使用英文，避免 Windows 中文终端（GBK）下 print 中文触发
    UnicodeEncodeError；
  * 默认开启 AMP（4060 支持 TF32/FP16），显存不足时降低 --batch 或 --crop；
  * 验证用 batch=1（原图长宽比不一，无法堆叠）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pestcount.dataset import PointCountDataset, collate, load_index
from pestcount.loss import DensityCountLoss
from pestcount.metrics import count_metrics, summarize
from pestcount.model import build_model, count_parameters
from pestcount.predictor import load_checkpoint
from pestcount.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Point-supervised pest counting")
    p.add_argument("--data-root", type=str, default="data/images")
    p.add_argument("--train-json", type=str, default="data/annotations/train.json")
    p.add_argument("--val-json", type=str, default="data/annotations/val.json")
    p.add_argument("--out", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--infer-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--width", type=float, default=1.5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lambda-local", type=float, default=1.0)
    p.add_argument("--grid", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="debug: 只取前 N 条样本")
    p.add_argument("--init-weights", type=str, default="",
                   help="从已有权重热启动（两阶段训练第二步：合成预训练 -> 真实微调）")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def build_loaders(args: argparse.Namespace):
    train_items = load_index(args.train_json)["items"]
    val_items = load_index(args.val_json)["items"]
    if args.limit > 0:
        train_items, val_items = train_items[:args.limit], val_items[:max(2, args.limit // 4)]

    train_ds = PointCountDataset(train_items, args.data_root, mode="train",
                                 crop_size=args.crop, stride=4,
                                 infer_size=args.infer_size)
    val_ds = PointCountDataset(val_items, args.data_root, mode="val",
                               stride=4, infer_size=args.infer_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, drop_last=len(train_ds) >= args.batch,
                              collate_fn=collate,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=lambda b: b[0])
    return train_loader, val_loader


def warmup_cosine(optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
    """线性 warmup + cosine 衰减。"""
    def fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def validate(model, loader, device) -> dict:
    model.eval()
    preds, gts = [], []
    for sample in loader:
        img = sample["image"].unsqueeze(0).to(device, non_blocking=True)
        pred = model(img)
        preds.append(float(pred.sum().item()))
        gts.append(float(sample["count"]))
    model.train()
    return count_metrics(preds, gts)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(args)
    model = build_model(width=args.width).to(device)
    if args.init_weights:
        blob = load_checkpoint(args.init_weights, device)
        model.load_state_dict(blob.get("model", blob))
        print(f"init weights from {args.init_weights}")
    n_param = count_parameters(model)
    print(f"device={device}  params={n_param/1e6:.3f}M  "
          f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")

    criterion = DensityCountLoss(lambda_local=args.lambda_local, grid=args.grid)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = warmup_cosine(optimizer, args.warmup, args.epochs, args.lr)
    amp_enabled = (not args.no_amp) and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:                                        # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "val_mae",
                                "val_rmse", "val_acc", "seconds"])

    best_mae = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        running, n_batch = 0.0, 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["density"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(images)
                loss, _ = criterion(pred, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.detach())
            n_batch += 1

        scheduler.step()
        val = validate(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        secs = time.time() - t0

        print(f"epoch {epoch+1:3d}/{args.epochs}  loss={running/max(1,n_batch):.4f}  "
              f"MAE={val['mae']:.3f}  RMSE={val['rmse']:.3f}  "
              f"Acc={val['acc']*100:.1f}%  lr={lr_now:.2e}  {secs:.1f}s", flush=True)

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch + 1, f"{lr_now:.6f}",
                                    f"{running/max(1,n_batch):.6f}",
                                    f"{val['mae']:.6f}", f"{val['rmse']:.6f}",
                                    f"{val['acc']:.6f}", f"{secs:.2f}"])

        payload = {
            "model": model.state_dict(),
            "args": vars(args),
            "metrics": val,
            "epoch": epoch + 1,
        }
        torch.save(payload, out_dir / "last.pt")
        if val["mae"] < best_mae:
            best_mae = val["mae"]
            torch.save(payload, out_dir / "best.pt")
            with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump({"best_mae": best_mae, "epoch": epoch + 1,
                           "metrics": val, "params": n_param}, f, indent=2)

    print(f"\nDONE  best_mae={best_mae:.3f}  checkpoint={out_dir/'best.pt'}")


if __name__ == "__main__":
    main()
