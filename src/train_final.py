"""
train_final.py -- simplified, self-contained trainer for the SEA food
classifier (YOLOv11 backbone + custom classification head).

Per team requirements this version:
  KEPT    - the model code (backbone extraction + custom head)
  KEPT    - error handling throughout (data loading, training, checkpointing)
  REMOVED - the data preparation extras (augmentation transforms, resume
            logic, LR scheduling). Images are only resized and normalized.

Usage:
    python train_final.py --data ../dataset --epochs 20
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- model ----
def load_yolov11_backbone(weights_path="yolo11n.pt", cutoff=11):
    """Layers 0-10 of YOLOv11 = the backbone; the detection head is discarded."""
    try:
        full_model = YOLO(weights_path).model.model
    except Exception as e:
        sys.exit(f"ERROR: could not load YOLO weights '{weights_path}': {e}")
    return nn.Sequential(*list(full_model[:cutoff]))


class FoodYOLOClassifier(nn.Module):
    """Pretrained YOLOv11 backbone + our own classification head."""

    def __init__(self, num_classes, weights_path="yolo11n.pt",
                 feature_channels=256, freeze_backbone=True,
                 unfreeze_last_n=3, dropout=0.3):
        super().__init__()
        self.backbone = load_yolov11_backbone(weights_path)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_channels, num_classes)

        for p in self.backbone.parameters():
            p.requires_grad = not freeze_backbone
        if freeze_backbone and unfreeze_last_n > 0:
            for layer in list(self.backbone.children())[-unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True

    def forward(self, x):
        feats = self.backbone(x)
        pooled = torch.flatten(self.pool(feats), 1)
        return self.classifier(self.dropout(pooled))


# ----------------------------------------------------------------- data ----
def get_loaders(root, batch_size):
    """Plain loaders: resize + normalize only (no augmentation)."""
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    loaders = {}
    for split in ("train", "val", "test"):
        path = os.path.join(root, split)
        if not os.path.isdir(path):
            sys.exit(f"ERROR: missing dataset split folder: {path}")
        try:
            ds = datasets.ImageFolder(path, transform=tf)
        except Exception as e:
            sys.exit(f"ERROR: could not read images in {path}: {e}")
        if len(ds) == 0:
            sys.exit(f"ERROR: no images found in {path}")
        loaders[split] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(split == "train"), num_workers=1)
    classes = loaders["train"].dataset.classes
    for split in ("val", "test"):
        if loaders[split].dataset.classes != classes:
            sys.exit(f"ERROR: class folders in {split}/ do not match train/")
    return loaders, classes


# ------------------------------------------------------------- training ----
def run_epoch(model, loader, criterion, optimizer, device, train):
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
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../dataset")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="../checkpoints/final_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, classes = get_loaders(args.data, args.batch_size)
    print(f"Classes ({len(classes)}): {classes}")

    model = FoodYOLOClassifier(num_classes=len(classes),
                               weights_path=args.weights).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=args.lr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_val = 0.0

    try:
        for epoch in range(1, args.epochs + 1):
            tl, ta = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
            vl, va = run_epoch(model, loaders["val"], criterion, optimizer, device, False)
            print(f"Epoch {epoch:02d}/{args.epochs} | "
                  f"train_loss={tl:.4f} acc={ta:.4f} | val_loss={vl:.4f} acc={va:.4f}")
            if va > best_val:
                best_val = va
                try:
                    torch.save({"model_state_dict": model.state_dict(),
                                "classes": classes, "val_acc": va,
                                "epoch": epoch}, args.out)
                    print(f"  -> saved best (val_acc={va:.4f})")
                except OSError as e:
                    print(f"  WARNING: checkpoint save failed: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted -- best checkpoint so far is preserved on disk.")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            sys.exit("ERROR: GPU out of memory. Lower --batch_size and retry.")
        raise

    # final test evaluation with the best saved weights
    try:
        ckpt = torch.load(args.out, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    except FileNotFoundError:
        sys.exit("ERROR: no checkpoint was saved -- training never improved.")
    _, test_acc = run_epoch(model, loaders["test"], criterion, optimizer, device, False)
    print(f"\nBest val acc: {best_val:.4f} | TEST acc: {test_acc:.4f}")
    print(f"Checkpoint: {args.out}")


if __name__ == "__main__":
    main()
