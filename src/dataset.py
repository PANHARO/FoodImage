"""
dataset.py

Loads the SEA food dataset from the standard ImageFolder layout:

    dataset/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg
        test/<class_name>/*.jpg

Written manually (not just a one-line torchvision.datasets.ImageFolder call)
so augmentation and normalization are explicit and easy to adjust.
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def get_dataloaders(dataset_root: str, batch_size: int = 32, num_workers: int = 2):
    """
    dataset_root should contain train/, val/, test/ subfolders,
    each with one subfolder per class.
    """
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")
    test_dir = os.path.join(dataset_root, "test")

    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(train=True))
    val_ds = datasets.ImageFolder(val_dir, transform=build_transforms(train=False))
    test_ds = datasets.ImageFolder(test_dir, transform=build_transforms(train=False))

    # class_to_idx must match across splits -- ImageFolder sorts class names
    # alphabetically so this holds as long as all three folders contain the
    # same set of class subfolders.
    assert train_ds.classes == val_ds.classes == test_ds.classes, (
        "Class folders differ between train/val/test splits! "
        f"train={train_ds.classes} val={val_ds.classes} test={test_ds.classes}"
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.classes


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "../dataset"
    train_loader, val_loader, test_loader, classes = get_dataloaders(root, batch_size=8)
    print("classes:", classes)
    xb, yb = next(iter(train_loader))
    print("batch shape:", xb.shape, "labels:", yb[:8])
