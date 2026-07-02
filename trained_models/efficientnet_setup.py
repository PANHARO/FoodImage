"""
SEA Food Recognition — EfficientNet-B0 Training Setup
======================================================
Run in order:
  1. python efficientnet_setup.py --clean      # fix dataset issues
  2. python efficientnet_setup.py --train      # train the model
  3. python efficientnet_setup.py --evaluate   # evaluate on test set

Requirements:
  pip install torch torchvision efficientnet-pytorch pillow numpy scikit-learn
"""

import os
import sys
import argparse
import shutil
import numpy as np
from PIL import Image
from collections import Counter

# ─────────────────────────────────────────────
# STEP 1 — DATASET CLEANUP
# ─────────────────────────────────────────────

# 46 near-duplicate images that appear in BOTH train AND val/test
# These cause data leakage — the model sees test images during training,
# inflating accuracy scores artificially. Remove them from val/test.
LEAKAGE_FILES = [
    "dataset/test/adobo/adobo_abdd95cb37b4.jpg",
    "dataset/test/adobo/adobo_ae8b86621894.jpg",
    "dataset/test/amok_trey/amok_trey_99ef82ac4c6a.jpg",
    "dataset/test/banh_mi/banh_mi_0992c0b06c66.jpg",
    "dataset/test/banh_mi/banh_mi_437420f8110d.jpg",
    "dataset/test/banh_mi/banh_mi_89e6cf838ea7.jpg",
    "dataset/test/hainanese_chicken_rice/hainanese_chicken_rice_3615b7601030.jpg",
    "dataset/test/laksa/laksa_8c2a2771bc34.jpg",
    "dataset/test/mohinga/mohinga_1628064d4667.jpg",
    "dataset/test/mohinga/mohinga_3863723a8e75.jpg",
    "dataset/test/mohinga/mohinga_4617f4f0cbac.jpg",
    "dataset/test/mohinga/mohinga_5b4f3e47d4cc.jpg",
    "dataset/test/mohinga/mohinga_5fe138f6a5bf.jpg",
    "dataset/test/mohinga/mohinga_7ea01b86722d.jpg",
    "dataset/test/mohinga/mohinga_81089515625d.jpg",
    "dataset/test/mohinga/mohinga_a9ce4eaba220.jpg",
    "dataset/test/mohinga/mohinga_c0b3245b8596.jpg",
    "dataset/test/mohinga/mohinga_e5f655fd2d7a.jpg",
    "dataset/test/mohinga/mohinga_f761182f517c.jpg",
    "dataset/test/nasi_goreng/nasi_goreng_b02dfa4646c1.jpg",
    "dataset/test/pho/pho_20dc35e191bc.jpg",
    "dataset/test/pho/pho_dbe272b6380d.jpg",
    "dataset/test/pho/pho_f96e2f341a67.jpg",
    "dataset/val/amok_trey/amok_trey_eb274b595ec5.jpg",
    "dataset/val/banh_mi/banh_mi_0b644f7d9fed.jpg",
    "dataset/val/banh_mi/banh_mi_123a31f01014.jpg",
    "dataset/val/banh_mi/banh_mi_975c88a0dffd.jpg",
    "dataset/val/banh_mi/banh_mi_a90240488105.jpg",
    "dataset/val/laksa/laksa_c1eba836f941.jpg",
    "dataset/val/laksa/laksa_d1d7a7587da3.jpg",
    "dataset/val/laksa/laksa_f6d7a069b52d.jpg",
    "dataset/val/mohinga/mohinga_32e0fc485e8b.jpg",
    "dataset/val/mohinga/mohinga_5c27810f2c98.jpg",
    "dataset/val/mohinga/mohinga_618f997a817b.jpg",
    "dataset/val/mohinga/mohinga_71fb805e8803.jpg",
    "dataset/val/mohinga/mohinga_7277bc9fe691.jpg",
    "dataset/val/mohinga/mohinga_8161fd4770e1.jpg",
    "dataset/val/mohinga/mohinga_af38319055ab.jpg",
    "dataset/val/mohinga/mohinga_b8ad819e5f60.jpg",
    "dataset/val/mohinga/mohinga_e6e05b3f21e9.jpg",
    "dataset/val/mohinga/mohinga_eb534820bb3b.jpg",
    "dataset/val/nasi_goreng/nasi_goreng_b3c3158104fb.jpg",
    "dataset/val/nasi_goreng/nasi_goreng_dfae3af212b8.jpg",
    "dataset/val/pad_thai/pad_thai_e05d2934c1f7.jpg",
    "dataset/val/pho/pho_3a4a53f11f7f.jpg",
    "dataset/val/pho/pho_8a88e3b04e61.jpg",
]

# 27 near-grayscale images — these are B&W or desaturated photos that
# lack color cues EfficientNet relies on for food distinction.
# Review these manually; remove ones that are clearly B&W.
GRAYSCALE_IMAGES = [
    "dataset/train/satay/satay_37a68e8725ee.jpg",
    "dataset/train/satay/satay_d0641bb33812.jpg",
    "dataset/train/satay/satay_ba4a7632c400.jpg",
    "dataset/train/adobo/adobo_5fb53cfbb55a.jpg",
    "dataset/train/adobo/adobo_2fece1e71202.jpg",
    "dataset/train/hainanese_chicken_rice/hainanese_chicken_rice_604ce4e1876f.jpg",
    "dataset/train/hainanese_chicken_rice/hainanese_chicken_rice_aed358bb97eb.jpg",
    "dataset/train/hainanese_chicken_rice/hainanese_chicken_rice_4202b0bb9e9d.jpg",
    "dataset/train/hainanese_chicken_rice/hainanese_chicken_rice_73c5655afde7.jpg",
    "dataset/train/amok_trey/amok_trey_2851bc5a9e7f.jpg",
    "dataset/train/pho/pho_44db821de280.jpg",
    "dataset/train/pho/pho_49a15389837d.jpg",
    "dataset/train/pho/pho_ce2301a98703.jpg",
    "dataset/train/pad_thai/pad_thai_f5a4c145db1d.jpg",
    "dataset/train/mohinga/mohinga_1956c10fbd9f.jpg",
    "dataset/train/mohinga/mohinga_5a12d386364a.jpg",
    "dataset/train/mohinga/mohinga_338dd71f33d6.jpg",
    "dataset/train/nasi_goreng/nasi_goreng_b54af897bb4f.jpg",
    "dataset/train/nasi_goreng/nasi_goreng_38088b53ac5c.jpg",
    "dataset/train/nasi_goreng/nasi_goreng_bfccae219084.jpg",
    "dataset/train/nasi_goreng/nasi_goreng_faca4c925191.jpg",
    "dataset/val/satay/satay_af7ce0d8f51c.jpg",
    "dataset/val/mohinga/mohinga_b2c1f167f9c3.jpg",
    "dataset/val/mohinga/mohinga_618f997a817b.jpg",
    "dataset/test/hainanese_chicken_rice/hainanese_chicken_rice_1902a38844bd.jpg",
    "dataset/test/mohinga/mohinga_3eefb1a5b2c4.jpg",
    "dataset/test/mohinga/mohinga_81067122e8b4.jpg",
]

# 1 cross-class mislabel — visually identical image in two different classes
# Inspect both manually and delete whichever is incorrectly labeled
MISLABELS = [
    ("dataset/train/adobo/adobo_0d58d56e926e.jpg",
     "dataset/train/laksa/laksa_d76548fc9d4d.jpg"),
]


def run_cleanup():
    print("=" * 55)
    print("STEP 1: DATASET CLEANUP")
    print("=" * 55)

    # --- Fix 1: Data leakage ---
    removed = 0
    for f in LEAKAGE_FILES:
        if os.path.exists(f):
            os.remove(f)
            removed += 1
    print(f"[1/3] Leakage: removed {removed}/{len(LEAKAGE_FILES)} duplicate val/test files")

    # --- Fix 2: Grayscale images ---
    print(f"\n[2/3] Near-grayscale images ({len(GRAYSCALE_IMAGES)} found):")
    print("      These lack color info. Review and delete the B&W ones.")
    print("      Files to inspect:")
    for f in GRAYSCALE_IMAGES:
        exists = "EXISTS" if os.path.exists(f) else "missing"
        print(f"        [{exists}] {f}")

    # --- Fix 3: Cross-class mislabel ---
    print(f"\n[3/3] Cross-class mislabel (1 pair found — inspect visually):")
    for a, b in MISLABELS:
        print(f"        {a}")
        print(f"        {b}")
        print("      Open both images and delete whichever is wrongly labeled.")

    # --- Final counts ---
    print("\n--- Post-cleanup dataset counts ---")
    for split in ['train', 'val', 'test']:
        total = 0
        for cls in sorted(os.listdir(f'dataset/{split}')):
            cls_path = f'dataset/{split}/{cls}'
            if os.path.isdir(cls_path):
                n = len([f for f in os.listdir(cls_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                total += n
        print(f"  {split}: {total} images")
    print("\nCleanup complete.\n")


# ─────────────────────────────────────────────
# STEP 2 — TRAINING
# ─────────────────────────────────────────────

# Dataset-specific normalization values (computed from YOUR train set)
# R mean is notably higher than ImageNet (0.559 vs 0.485) because food
# images are warmer/brighter than general ImageNet content.
DATASET_MEAN = [0.5588, 0.4799, 0.3743]
DATASET_STD  = [0.2509, 0.2472, 0.2625]

CLASSES = [
    "adobo", "amok_trey", "banh_mi", "hainanese_chicken_rice",
    "laksa", "mohinga", "nasi_goreng", "pad_thai", "pho", "satay"
]

# Training hyperparameters
BATCH_SIZE    = 32
LR_HEAD       = 1e-3   # Phase 1: only the new classifier head
LR_FINETUNE   = 1e-4   # Phase 2: unfrozen last blocks
EPOCHS_PHASE1 = 5      # Train head with frozen backbone
EPOCHS_PHASE2 = 25     # Fine-tune with unfrozen last 3 blocks
WEIGHT_DECAY  = 1e-4


def get_device(require_cuda=False):
    import torch
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please run this script on a machine with a GPU."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device}. GPU not detected.")
    return device


def run_training():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, transforms, models
    from torch.optim.lr_scheduler import CosineAnnealingLR

    device = get_device(require_cuda=True)

    # ── Transforms ──────────────────────────────────────────────────────
    # Images are already 224x224 so no resize needed in eval.
    # Training uses aggressive augmentation because we only have ~300/class.
    train_transforms = transforms.Compose([
        # Geometric augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=20),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),  # simulate zoom
        # Color augmentations — important for food (lighting varies a lot)
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.3,
            saturation=0.3,
            hue=0.08          # subtle hue shift; too high confuses food colors
        ),
        transforms.RandomGrayscale(p=0.03),  # rare; teaches robustness
        transforms.ToTensor(),
        # Use YOUR dataset's actual mean/std, not ImageNet defaults
        # Your R channel is notably warmer (0.559 vs ImageNet's 0.485)
        transforms.Normalize(DATASET_MEAN, DATASET_STD),
    ])

    eval_transforms = transforms.Compose([
        # No augmentation — images are already 224x224
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD),
    ])

    # ── Datasets ────────────────────────────────────────────────────────
    train_ds = datasets.ImageFolder("dataset/train", transform=train_transforms)
    val_ds   = datasets.ImageFolder("dataset/val",   transform=eval_transforms)
    test_ds  = datasets.ImageFolder("dataset/test",  transform=eval_transforms)

    print(f"Classes: {train_ds.classes}")

    # ── Weighted sampler for class imbalance ────────────────────────────
    # satay has 1.42x more images than amok_trey — sampler equalizes this
    class_counts = Counter(train_ds.targets)
    weights = [1.0 / class_counts[t] for t in train_ds.targets]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(train_ds),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────
    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace classifier head for 10 classes
    # Original: Linear(1280 -> 1000). Replace with dropout + linear for 10.
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 10)
    )
    model = model.to(device)

    # ── Loss ─────────────────────────────────────────────────────────────
    # WeightedRandomSampler already handles class imbalance in sampling,
    # but adding class weights to the loss adds a second layer of robustness
    class_weight_vals = torch.tensor(
        [1.0 / class_counts[i] for i in range(10)], dtype=torch.float
    )
    class_weight_vals = class_weight_vals / class_weight_vals.sum() * 10  # normalize
    criterion = nn.CrossEntropyLoss(weight=class_weight_vals.to(device))

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1: Freeze backbone, train new head only (5 epochs)
    # This lets the new classifier stabilize before touching the pretrained
    # weights, which prevents the backbone from being corrupted early on.
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("PHASE 1: Training classifier head (backbone frozen)")
    print("=" * 55)

    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE1)

    best_val_acc = 0.0
    for epoch in range(EPOCHS_PHASE1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        val_acc = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch+1}/{EPOCHS_PHASE1} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Train acc: {correct/total:.3f} | Val acc: {val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2: Unfreeze last 3 blocks of backbone + fine-tune
    # Only unfreeze the last 3 blocks (not all), because with ~300 images/
    # class fully unfreezing risks catastrophic forgetting of ImageNet
    # features that still help with texture, color, and shape recognition.
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("PHASE 2: Fine-tuning (last 3 backbone blocks unfrozen)")
    print("=" * 55)

    # Unfreeze last 3 feature blocks
    blocks_to_unfreeze = list(model.features.children())[-3:]
    for block in blocks_to_unfreeze:
        for param in block.parameters():
            param.requires_grad = True

    # Use a lower LR for fine-tuning to protect pretrained weights
    optimizer = torch.optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': LR_FINETUNE},
        {'params': [p for b in blocks_to_unfreeze
                    for p in b.parameters() if p.requires_grad],
         'lr': LR_FINETUNE * 0.1}   # backbone gets 10x lower LR than head
    ], weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE2)

    for epoch in range(EPOCHS_PHASE2):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        val_acc = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch+1}/{EPOCHS_PHASE2} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Train acc: {correct/total:.3f} | Val acc: {val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"    ✓ Saved new best model (val acc: {val_acc:.3f})")

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    print("Model saved to: best_model.pth")


def evaluate(model, loader, device):
    import torch
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


# ─────────────────────────────────────────────
# STEP 3 — EVALUATION
# ─────────────────────────────────────────────

def run_evaluation():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models
    from sklearn.metrics import classification_report, confusion_matrix

    device = get_device(require_cuda=False)

    eval_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD),
    ])
    test_ds = datasets.ImageFolder("dataset/test", transform=eval_transforms)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # Load best model
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 10)
    )
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(all_labels, all_preds)
    print("        " + "  ".join(f"{c[:4]:>6}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:<12}" + "  ".join(f"{v:>6}" for v in row))


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean",    action="store_true", help="Clean dataset issues")
    parser.add_argument("--train",    action="store_true", help="Train EfficientNet-B0")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on test set")
    args = parser.parse_args()

    if args.clean:
        run_cleanup()
    elif args.train:
        run_training()
    elif args.evaluate:
        run_evaluation()
    else:
        print("Usage:")
        print("  python efficientnet_setup.py --clean      # fix dataset issues first")
        print("  python efficientnet_setup.py --train      # then train")
        print("  python efficientnet_setup.py --evaluate   # then evaluate")
