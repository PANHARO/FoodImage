"""Evaluate a saved EfficientNet-B0 checkpoint."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from config import (  # type: ignore
        BATCH_SIZE, CLASSES, DATA_ROOT, MODEL_NAME, MODEL_PATH, NUM_WORKERS,
    )
    from feature_extraction import build_model, get_device  # type: ignore
    from preprocessing import build_transforms, validate_classes  # type: ignore
else:
    from .config import (
        BATCH_SIZE, CLASSES, DATA_ROOT, MODEL_NAME, MODEL_PATH, NUM_WORKERS,
    )
    from .feature_extraction import build_model, get_device
    from .preprocessing import build_transforms, validate_classes

try:
    from metrics import save_evaluation_plots
except ImportError:
    from trained_models.metrics import save_evaluation_plots


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
    save_evaluation_plots(labels, predictions, CLASSES, MODEL_NAME)


if __name__ == "__main__":
    run_evaluation()
