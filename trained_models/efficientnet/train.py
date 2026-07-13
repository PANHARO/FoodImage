"""Train EfficientNet-B0 on the Southeast Asian food dataset."""

import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from config import (  # type: ignore
        BATCH_SIZE, CLASSES, DATA_ROOT, EPOCHS_PHASE1, EPOCHS_PHASE2,
        LR_FINETUNE, LR_HEAD, MODEL_NAME, MODEL_PATH, NUM_WORKERS,
        WEIGHT_DECAY,
    )
    from feature_extraction import (  # type: ignore
        build_model, freeze_feature_extractor, get_device,
        unfreeze_final_feature_blocks,
    )
    from preprocessing import build_transforms, validate_classes  # type: ignore
else:
    from .config import (
        BATCH_SIZE, CLASSES, DATA_ROOT, EPOCHS_PHASE1, EPOCHS_PHASE2,
        LR_FINETUNE, LR_HEAD, MODEL_NAME, MODEL_PATH, NUM_WORKERS,
        WEIGHT_DECAY,
    )
    from .feature_extraction import (
        build_model, freeze_feature_extractor, get_device,
        unfreeze_final_feature_blocks,
    )
    from .preprocessing import build_transforms, validate_classes

try:
    from metrics import save_training_history
except ImportError:
    from trained_models.metrics import save_training_history


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


def run_training():
    """Train EfficientNet-B0 in head-training and fine-tuning phases."""
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

    model = build_model(models.EfficientNet_B0_Weights.IMAGENET1K_V1).to(device)
    # Class weights make mistakes on small classes count proportionally more.
    class_weights = torch.tensor(
        [1.0 / class_counts[index] for index in range(len(CLASSES))],
        dtype=torch.float32,
        device=device,
    )
    class_weights = class_weights / class_weights.sum() * len(CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history = []

    freeze_feature_extractor(model)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    best_validation_accuracy = train_phase(
        model, train_loader, validation_loader, criterion, optimizer,
        CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE1), device,
        EPOCHS_PHASE1, "Phase 1: classifier head", -1.0, history,
    )

    blocks_to_unfreeze = unfreeze_final_feature_blocks(model)
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
    save_training_history(history, MODEL_NAME)
    print(f"\nBest validation accuracy: {best_validation_accuracy:.3f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    run_training()
