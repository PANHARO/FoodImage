"""
train.py

Manual training loop (no ultralytics model.train(), no high-level
Trainer class) for the FoodYOLOClassifier defined in model.py.

Usage:
    python train.py --data ../dataset --epochs 15 --freeze
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_dataloaders
from model import FoodYOLOClassifier


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    torch.set_grad_enabled(True)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../dataset", help="path to dataset root (train/val/test)")
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze", action="store_true", default=True, help="freeze backbone, train head only")
    parser.add_argument("--unfreeze_last_n", type=int, default=0, help="also fine-tune last N backbone layers")
    parser.add_argument("--out_dir", type=str, default="../checkpoints")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs w/o val improvement)")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoints/best_model.pt if present")
    parser.add_argument("--init_from", type=str, default=None, help="load weights from another checkpoint as a starting point (fresh optimizer/history) -- used for phase-2 fine-tuning")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.data, batch_size=args.batch_size, num_workers=1
    )
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    model = FoodYOLOClassifier(
        num_classes=num_classes,
        weights_path=args.weights,
        freeze_backbone=args.freeze,
        unfreeze_last_n=args.unfreeze_last_n,
    ).to(device)
    print(f"Trainable params: {model.trainable_parameter_count():,} / {model.total_parameter_count():,}")

    if args.init_from and os.path.exists(args.init_from):
        init_ckpt = torch.load(args.init_from, map_location=device)
        model.load_state_dict(init_ckpt["model_state_dict"])
        print(f"Initialized weights from {args.init_from} (val_acc was {init_ckpt.get('val_acc', 0):.4f})")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0
    epochs_no_improve = 0
    history = []
    start_epoch = 1

    history_path = os.path.join(args.out_dir, "history.json")
    ckpt_path = os.path.join(args.out_dir, "best_model.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_acc = ckpt.get("val_acc", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
        print(f"Resumed from epoch {ckpt.get('epoch', 0)} (best_val_acc={best_val_acc:.4f}); continuing from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_acc)
        dt = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {dt:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "seconds": dt,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "val_acc": val_acc,
                "epoch": epoch,
            }, ckpt_path)
            print(f"  -> new best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
            break

    with open(os.path.join(args.out_dir, "classes.json"), "w") as f:
        json.dump(classes, f, indent=2)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {os.path.join(args.out_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
