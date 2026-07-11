"""
train_best.py -- "best result" configuration, intended for GPU (Colab).

Upgrades over train.py:
  - yolo11s backbone (512-ch features) instead of yolo11n
  - two-phase schedule in one run: short head warmup, then FULL backbone
    fine-tune with discriminative learning rates (backbone lower than head)
  - label smoothing, cosine LR schedule, RandAugment, mixed precision (AMP)
  - saves best-by-val checkpoint and runs final test evaluation

Usage (GPU strongly recommended):
    python train_best.py --data ../dataset
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import FoodYOLOClassifier

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_loaders(root, batch_size, num_workers):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.15),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tr = datasets.ImageFolder(os.path.join(root, "train"), train_tf)
    va = datasets.ImageFolder(os.path.join(root, "val"), eval_tf)
    te = datasets.ImageFolder(os.path.join(root, "test"), eval_tf)
    assert tr.classes == va.classes == te.classes
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        tr.classes,
    )


def run_epoch(model, loader, criterion, optimizer, scaler, device, train):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out = model(x)
            loss = criterion(out, y)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../dataset")
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--feature_channels", type=int, default=512)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--finetune_epochs", type=int, default=45)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--backbone_lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--out_dir", default="../checkpoints_best")
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU detected. This config is slow on CPU -- use Colab (Runtime > Change runtime type > GPU).")
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, val_loader, test_loader, classes = build_loaders(
        args.data, args.batch_size, args.num_workers
    )
    print(f"Device: {device} | Classes: {classes}")

    model = FoodYOLOClassifier(
        num_classes=len(classes),
        weights_path=args.weights,
        feature_channels=args.feature_channels,
        freeze_backbone=True,  # phase 1: head warmup
        dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    ckpt_path = os.path.join(args.out_dir, "best_model.pt")
    history, best_val = [], 0.0

    # ---------- Phase 1: head warmup ----------
    opt = AdamW(model.classifier.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.warmup_epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, criterion, opt, scaler, device, True)
        vl, va = run_epoch(model, val_loader, criterion, opt, scaler, device, False)
        print(f"[warmup {epoch}/{args.warmup_epochs}] train_acc={ta:.4f} val_acc={va:.4f} ({time.time()-t0:.0f}s)")
        history.append({"phase": "warmup", "epoch": epoch, "train_acc": ta, "val_acc": va,
                        "train_loss": tl, "val_loss": vl})

    # ---------- Phase 2: full fine-tune, discriminative LRs ----------
    for prm in model.backbone.parameters():
        prm.requires_grad = True
    opt = AdamW([
        {"params": model.backbone.parameters(), "lr": args.backbone_lr},
        {"params": model.classifier.parameters(), "lr": args.head_lr * 0.1},
    ], weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.finetune_epochs)

    no_improve = 0
    for epoch in range(1, args.finetune_epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, criterion, opt, scaler, device, True)
        vl, va = run_epoch(model, val_loader, criterion, opt, scaler, device, False)
        sched.step()
        print(f"[finetune {epoch}/{args.finetune_epochs}] train_acc={ta:.4f} val_acc={va:.4f} ({time.time()-t0:.0f}s)")
        history.append({"phase": "finetune", "epoch": epoch, "train_acc": ta, "val_acc": va,
                        "train_loss": tl, "val_loss": vl})
        if va > best_val:
            best_val = va
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "classes": classes,
                        "val_acc": va, "epoch": epoch,
                        "weights": args.weights, "feature_channels": args.feature_channels},
                       ckpt_path)
            print(f"  -> saved best (val_acc={va:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("Early stopping.")
                break
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # ---------- Final: evaluate best checkpoint on test ----------
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    _, test_acc = run_epoch(model, test_loader, criterion, opt, scaler, device, False)
    print(f"\nBest val acc: {best_val:.4f}")
    print(f"TEST accuracy (best ckpt): {test_acc:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print("Run evaluate.py for full precision/recall/F1 + confusion matrix "
          f"(pass --ckpt {ckpt_path} and --unfreeze_last_n 11).")


if __name__ == "__main__":
    main()
