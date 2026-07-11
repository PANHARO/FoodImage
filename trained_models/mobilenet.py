"""Train and evaluate MobileNetV2 on the food dataset."""

import argparse
from collections import Counter
from pathlib import Path

try:
    from .metrics import save_evaluation_plots, save_training_history
except ImportError:  # Support: python trained_models/mobilenet.py
    from metrics import save_evaluation_plots, save_training_history

DATA_ROOT = Path("dataset")
MODEL_PATH = Path("trained_models/checkpoints/best_mobilenet_v2_model.pth")
DATASET_MEAN = [0.5588, 0.4799, 0.3743]
DATASET_STD = [0.2509, 0.2472, 0.2625]
CLASSES = [
    "adobo", "amok_trey", "banh_mi", "hainanese_chicken_rice", "laksa",
    "laphet_thoke", "nasi_goreng", "pad_thai", "pho", "satay",
]

BATCH_SIZE = 32
NUM_WORKERS = 4
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 25
WEIGHT_DECAY = 1e-4


def get_device(require_cuda=False):
    import torch

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build "
            "and confirm that the NVIDIA driver is working."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device}. GPU not detected.")
    return device


def build_transforms():
    from torchvision import transforms

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=20),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.3, saturation=0.3, hue=0.08
        ),
        transforms.RandomGrayscale(p=0.03),
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD),
    ])
    eval_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD),
    ])
    return train_transforms, eval_transforms


def validate_classes(*datasets):
    for dataset in datasets:
        if dataset.classes != CLASSES:
            raise RuntimeError(
                f"Unexpected classes in {dataset.root}: {dataset.classes}. "
                f"Expected: {CLASSES}"
            )


def build_model(weights):
    import torch.nn as nn
    from torchvision import models

    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))
    return model


def evaluate_accuracy(model, loader, device):
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def train_phase(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, epochs, phase_name, best_val_acc, history):
    import torch

    print("\n" + "=" * 55)
    print(phase_name)
    print("=" * 55)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        val_acc = evaluate_accuracy(model, val_loader, device)
        train_acc = correct / total if total else 0.0
        average_loss = total_loss / len(train_loader)
        history.append({
            "epoch": len(history) + 1,
            "phase": phase_name.split(":", 1)[0],
            "loss": average_loss,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
        })
        print(
            f"  Epoch {epoch + 1}/{epochs} | "
            f"Loss: {average_loss:.4f} | "
            f"Train acc: {train_acc:.3f} | Val acc: {val_acc:.3f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"    Saved new best model (val acc: {val_acc:.3f})")
    return best_val_acc


def run_training():
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, models

    # Prefer CUDA, but allow training on CPU when no compatible GPU is found.
    device = get_device(require_cuda=False)
    train_transforms, eval_transforms = build_transforms()
    train_dataset = datasets.ImageFolder(
        DATA_ROOT / "train", transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        DATA_ROOT / "val", transform=eval_transforms
    )
    test_dataset = datasets.ImageFolder(
        DATA_ROOT / "test", transform=eval_transforms
    )
    validate_classes(train_dataset, val_dataset, test_dataset)
    print(f"Classes: {train_dataset.classes}")

    class_counts = Counter(train_dataset.targets)
    sample_weights = [
        1.0 / class_counts[target] for target in train_dataset.targets
    ]
    sampler = WeightedRandomSampler(
        torch.DoubleTensor(sample_weights), len(train_dataset), replacement=True
    )
    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = build_model(models.MobileNet_V2_Weights.IMAGENET1K_V2).to(device)
    class_weights = torch.tensor(
        [1.0 / class_counts[index] for index in range(len(CLASSES))],
        dtype=torch.float32,
        device=device,
    )
    class_weights = class_weights / class_weights.sum() * len(CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Phase 1: freeze the backbone and train the classifier.
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE1)
    history = []
    best_val_acc = train_phase(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, EPOCHS_PHASE1,
        "PHASE 1: Training classifier head (backbone frozen)", -1.0, history,
    )

    # Phase 2: fine-tune the final three MobileNetV2 feature blocks.
    blocks_to_unfreeze = list(model.features.children())[-3:]
    for block in blocks_to_unfreeze:
        for parameter in block.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": LR_FINETUNE},
        {
            "params": [
                parameter
                for block in blocks_to_unfreeze
                for parameter in block.parameters()
                if parameter.requires_grad
            ],
            "lr": LR_FINETUNE * 0.1,
        },
    ], weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE2)
    best_val_acc = train_phase(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, EPOCHS_PHASE2,
        "PHASE 2: Fine-tuning last 3 backbone blocks", best_val_acc, history,
    )
    save_training_history(history, "mobilenet_v2")
    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {MODEL_PATH}")


def run_evaluation():
    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    from torch.utils.data import DataLoader
    from torchvision import datasets

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} does not exist. Train MobileNetV2 first."
        )
    device = get_device(require_cuda=False)
    _, eval_transforms = build_transforms()
    test_dataset = datasets.ImageFolder(
        DATA_ROOT / "test", transform=eval_transforms
    )
    validate_classes(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model = build_model(weights=None)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            predictions = model(images).argmax(dim=1).cpu().tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels.tolist())

    label_ids = list(range(len(CLASSES)))
    print("\n=== Classification Report ===")
    print(classification_report(
        all_labels, all_predictions, labels=label_ids,
        target_names=CLASSES, zero_division=0,
    ))
    print("=== Confusion Matrix ===")
    matrix = confusion_matrix(all_labels, all_predictions, labels=label_ids)
    print("        " + "  ".join(f"{name[:4]:>6}" for name in CLASSES))
    for class_name, row in zip(CLASSES, matrix):
        print(f"{class_name:<12}" + "  ".join(f"{value:>6}" for value in row))
    save_evaluation_plots(all_labels, all_predictions, CLASSES, "mobilenet_v2")


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true", help="Train MobileNetV2")
    action.add_argument(
        "--evaluate", action="store_true", help="Evaluate the saved model"
    )
    args = parser.parse_args()
    if args.train:
        run_training()
    else:
        run_evaluation()


if __name__ == "__main__":
    main()
