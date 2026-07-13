"""Preprocessing and dataset validation for MobileNetV2."""

try:
    from .config import CLASSES, DATASET_MEAN, DATASET_STD
except ImportError:  # Support running files in this folder directly.
    from config import CLASSES, DATASET_MEAN, DATASET_STD


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


def validate_classes(*datasets):
    """Fail early if a split is missing, renamed, or reordered classes."""
    for dataset in datasets:
        if dataset.classes != CLASSES:
            raise RuntimeError(
                f"Unexpected classes in {dataset.root}: {dataset.classes}. "
                f"Expected: {CLASSES}"
            )
