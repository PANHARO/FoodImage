"""Train and evaluate MobileNetV2 on the Southeast Asian food dataset.

Run from the project root:
    python trained_models/mobilenet.py --train
    python trained_models/mobilenet.py --evaluate
"""

import argparse
from collections import Counter
from pathlib import Path

try:
    from .metrics import save_evaluation_plots, save_training_history
except ImportError:  # Support running this file directly.
    from metrics import save_evaluation_plots, save_training_history


# Dataset layout and class order must match torchvision.datasets.ImageFolder.
DATA_ROOT = Path("dataset")
MODEL_PATH = Path("trained_models/checkpoints/best_mobilenet_v2_model.pth")
CLASSES = [
    "adobo", "amok_trey", "banh_mi", "hainanese_chicken_rice", "laksa",
    "laphet_thoke", "nasi_goreng", "pad_thai", "pho", "satay",
]

# Normalization statistics calculated from this project's training images.
DATASET_MEAN = [0.5588, 0.4799, 0.3743]
DATASET_STD = [0.2509, 0.2472, 0.2625]

# Training configuration: train the new head, then fine-tune final feature blocks.
BATCH_SIZE = 32
NUM_WORKERS = 4
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 25
WEIGHT_DECAY = 1e-4


# Select GPU acceleration when it is available.
def get_device():
    """Use the NVIDIA GPU when PyTorch can access it; otherwise use CPU."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu. GPU not detected.")
    return device


# Define image augmentation for training and normalization for evaluation.
def build_transforms():
    """Return augmented training transforms and deterministic evaluation transforms."""
    from torchvision import transforms

    train_transform = transforms.Compose([
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
    evaluation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD),
    ])
    return train_transform, evaluation_transform


# Ensure every dataset split has the expected class folders.
def validate_classes(*datasets):
    """Fail early if a split is missing, renamed, or reordered classes."""
    for dataset in datasets:
        if dataset.classes != CLASSES:
            raise RuntimeError(
                f"Unexpected classes in {dataset.root}: {dataset.classes}. "
                f"Expected: {CLASSES}"
            )


# Load MobileNetV2 and adapt its classifier for this dataset.
def build_model(weights):
    """Create MobileNetV2 and replace ImageNet's 1000-class classifier."""
    import torch.nn as nn
    from torchvision import models

    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))
    return model


# Calculate model accuracy without changing model weights.
def evaluate_accuracy(model, loader, device):
    """Return classification accuracy for a validation or test data loader."""
    import torch

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


# Train one phase and record its epoch statistics.
def train_phase(model, train_loader, validation_loader, criterion, optimizer,
                scheduler, device, epochs, phase_name, best_validation_accuracy,
                history):
    """Train one phase and append each epoch's loss and accuracies to history."""
    import torch

    print(f"\n{phase_name}")
    for epoch in range(epochs):
        model.train()
        total_loss = correct = total = 0
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
        loss = total_loss / len(train_loader)
        train_accuracy = correct / total if total else 0.0
        validation_accuracy = evaluate_accuracy(model, validation_loader, device)
        history.append({
            "epoch": len(history) + 1,
            "phase": phase_name,
            "loss": loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
        })
        print(
            f"  Epoch {epoch + 1}/{epochs} | Loss: {loss:.4f} | "
            f"Train acc: {train_accuracy:.3f} | "
            f"Val acc: {validation_accuracy:.3f}"
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"    Saved new best model (val acc: {validation_accuracy:.3f})")
    return best_validation_accuracy


# Run the complete two-phase MobileNet training workflow.
def run_training():
    """Train MobileNetV2 in head-training and fine-tuning phases."""
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, models

    device = get_device()
    train_transform, evaluation_transform = build_transforms()
    train_dataset = datasets.ImageFolder(DATA_ROOT / "train", train_transform)
    validation_dataset = datasets.ImageFolder(
        DATA_ROOT / "val", evaluation_transform
    )
    test_dataset = datasets.ImageFolder(DATA_ROOT / "test", evaluation_transform)
    validate_classes(train_dataset, validation_dataset, test_dataset)

    # The sampler gives underrepresented classes equal chances in each epoch.
    class_counts = Counter(train_dataset.targets)
    sample_weights = [1.0 / class_counts[target] for target in train_dataset.targets]
    sampler = WeightedRandomSampler(
        torch.DoubleTensor(sample_weights), len(train_dataset), replacement=True
    )
    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_options)
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )

    model = build_model(models.MobileNet_V2_Weights.IMAGENET1K_V2).to(device)
    # Class weights make mistakes on small classes count proportionally more.
    class_weights = torch.tensor(
        [1.0 / class_counts[index] for index in range(len(CLASSES))],
        dtype=torch.float32,
        device=device,
    )
    class_weights = class_weights / class_weights.sum() * len(CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history = []

    # Phase 1 trains only the new classifier while preserving pretrained features.
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    best_validation_accuracy = train_phase(
        model, train_loader, validation_loader, criterion, optimizer,
        CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE1), device,
        EPOCHS_PHASE1, "Phase 1: classifier head", -1.0, history,
    )

    # Phase 2 unfreezes only the final blocks for controlled fine-tuning.
    blocks_to_unfreeze = list(model.features.children())[-3:]
    for block in blocks_to_unfreeze:
        for parameter in block.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": LR_FINETUNE},
        {
            "params": [
                parameter for block in blocks_to_unfreeze
                for parameter in block.parameters() if parameter.requires_grad
            ],
            "lr": LR_FINETUNE * 0.1,
        },
    ], weight_decay=WEIGHT_DECAY)
    best_validation_accuracy = train_phase(
        model, train_loader, validation_loader, criterion, optimizer,
        CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE2), device,
        EPOCHS_PHASE2, "Phase 2: final feature blocks", best_validation_accuracy,
        history,
    )
    save_training_history(history, "mobilenet_v2")
    print(f"\nBest validation accuracy: {best_validation_accuracy:.3f}")
    print(f"Model saved to: {MODEL_PATH}")


# Evaluate the saved checkpoint and create performance reports.
def run_evaluation():
    """Evaluate the best checkpoint and save its confusion-matrix reports."""
    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    from torch.utils.data import DataLoader
    from torchvision import datasets

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} does not exist. Train first.")
    device = get_device()
    _, evaluation_transform = build_transforms()
    test_dataset = datasets.ImageFolder(DATA_ROOT / "test", evaluation_transform)
    validate_classes(test_dataset)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
    )
    model = build_model(weights=None).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    predictions, labels = [], []
    with torch.no_grad():
        for images, batch_labels in test_loader:
            predictions.extend(
                model(images.to(device, non_blocking=True)).argmax(dim=1).cpu().tolist()
            )
            labels.extend(batch_labels.tolist())

    label_ids = list(range(len(CLASSES)))
    print("\n=== Classification Report ===")
    print(classification_report(labels, predictions, labels=label_ids,
                                target_names=CLASSES, zero_division=0))
    print("=== Confusion Matrix ===")
    matrix = confusion_matrix(labels, predictions, labels=label_ids)
    for class_name, row in zip(CLASSES, matrix):
        print(f"{class_name:<24}" + " ".join(f"{value:>4}" for value in row))
    save_evaluation_plots(labels, predictions, CLASSES, "mobilenet_v2")


# Choose either training or evaluation from command-line arguments.
def main():
    """Parse a single requested action."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true", help="Train MobileNetV2")
    action.add_argument("--evaluate", action="store_true", help="Evaluate best checkpoint")
    args = parser.parse_args()
    if args.train:
        run_training()
    else:
        run_evaluation()


if __name__ == "__main__":
    main()
