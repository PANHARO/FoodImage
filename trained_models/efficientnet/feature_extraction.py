"""EfficientNet-B0 model construction and feature fine-tuning helpers."""

try:
    from .config import CLASSES
except ImportError:  # Support running files in this folder directly.
    from config import CLASSES


def get_device():
    """Use the NVIDIA GPU when PyTorch can access it; otherwise use CPU."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu. GPU not detected.")
    return device


def build_model(weights):
    """Create EfficientNet-B0 and replace ImageNet's 1000-class head."""
    import torch.nn as nn
    from torchvision import models

    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, len(CLASSES)),
    )
    return model


def freeze_feature_extractor(model):
    """Freeze pretrained feature layers so only the classifier trains."""
    for parameter in model.features.parameters():
        parameter.requires_grad = False


def unfreeze_final_feature_blocks(model, block_count=3):
    """Unfreeze the final feature blocks and return them for optimizer setup."""
    blocks_to_unfreeze = list(model.features.children())[-block_count:]
    for block in blocks_to_unfreeze:
        for parameter in block.parameters():
            parameter.requires_grad = True
    return blocks_to_unfreeze
