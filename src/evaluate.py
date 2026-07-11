"""
evaluate.py

Evaluates the trained checkpoint on the held-out test set and reports the
metrics promised in the proposal: accuracy, precision, recall, F1-score
(macro + per-class), plus a confusion matrix image and training curves.

Usage:
    python evaluate.py --data ../dataset --ckpt ../checkpoints/best_model.pt
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from dataset import get_dataloaders
from model import FoodYOLOClassifier


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (test set)")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history_path, out_path):
    if not os.path.exists(history_path):
        return
    with open(history_path) as f:
        history = json.load(f)
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.axvline(x=15.5, color="gray", ls="--", lw=1)
    ax1.text(15.8, ax1.get_ylim()[1]*0.95, "fine-tuning starts", fontsize=8, color="gray")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss"); ax1.legend()
    ax2.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], label="val")
    ax2.axvline(x=15.5, color="gray", ls="--", lw=1)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy"); ax2.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../dataset")
    parser.add_argument("--ckpt", type=str, default="../checkpoints/best_model.pt")
    parser.add_argument("--out_dir", type=str, default="../results")
    parser.add_argument("--unfreeze_last_n", type=int, default=3,
                        help="must match the value used in the final training phase")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["classes"]

    model = FoodYOLOClassifier(
        num_classes=len(classes),
        freeze_backbone=True,
        unfreeze_last_n=args.unfreeze_last_n,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, _, test_loader, loader_classes = get_dataloaders(args.data, batch_size=32, num_workers=1)
    assert loader_classes == classes, "Checkpoint classes don't match dataset folders!"

    y_true, y_pred = collect_predictions(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print("=" * 60)
    print(f"TEST SET RESULTS  ({len(y_true)} images)")
    print("=" * 60)
    print(f"Accuracy         : {acc:.4f}")
    print(f"Precision (macro): {prec_macro:.4f}")
    print(f"Recall (macro)   : {rec_macro:.4f}")
    print(f"F1-score (macro) : {f1_macro:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes, os.path.join(args.out_dir, "confusion_matrix.png"))
    plot_training_curves(os.path.join(os.path.dirname(args.ckpt), "history.json"),
                         os.path.join(args.out_dir, "training_curves.png"))

    report = classification_report(y_true, y_pred, target_names=classes,
                                   zero_division=0, output_dict=True)
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump({
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "per_class": report,
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_val_acc": ckpt.get("val_acc"),
        }, f, indent=2)

    print(f"Saved: confusion_matrix.png, training_curves.png, test_metrics.json -> {args.out_dir}")


if __name__ == "__main__":
    main()
