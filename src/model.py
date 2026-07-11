"""
model.py

Defines a custom image classifier built on top of a pretrained YOLOv11
backbone. This deliberately does NOT use ultralytics' built-in yolo11-cls
task or model.train() -- the backbone is extracted manually and the
classification head, forward pass, and training loop are all written by us
in plain PyTorch, per the assignment requirement to not depend on YOLO
end-to-end.

Architecture:
    YOLOv11n full model (from ultralytics, pretrained on COCO) has 24
    sequential layers. Layers 0-10 make up the backbone (stem + C3k2 blocks
    + SPPF + C2PSA attention block); layers 11-23 are the detection
    neck/head, which we discard entirely since we don't need bounding
    boxes -- this is a single-label classification task.

    Backbone(224x224x3) -> feature map (256, 7, 7)
    -> Global Average Pool -> (256,)
    -> Dropout -> Linear(256 -> num_classes)
"""

import torch
import torch.nn as nn
from ultralytics import YOLO


BACKBONE_CUTOFF = 11  # layers [0:11] = stem through C2PSA (see inspection below)
BACKBONE_OUT_CHANNELS = 256  # for yolo11n; see FEATURE_CHANNELS table below

# If you swap to a bigger YOLOv11 variant, the backbone output channel count
# changes. Pass the right value to FoodYOLOClassifier(feature_channels=...).
FEATURE_CHANNELS = {
    "yolo11n.pt": 256,
    "yolo11s.pt": 512,
    "yolo11m.pt": 576,
    "yolo11l.pt": 512,
    "yolo11x.pt": 512,
}


def load_yolov11_backbone(weights_path: str = "yolo11n.pt", cutoff: int = BACKBONE_CUTOFF):
    """
    Loads pretrained YOLOv11 detection weights and slices off everything
    after the backbone (the neck + detection head are discarded).

    Returns an nn.Sequential of the backbone layers only.
    """
    full_model = YOLO(weights_path).model.model  # nn.Sequential of 24 layers
    backbone_layers = list(full_model[:cutoff])
    backbone = nn.Sequential(*backbone_layers)
    return backbone


class FoodYOLOClassifier(nn.Module):
    """
    Custom classifier: pretrained YOLOv11 backbone + our own head.

    freeze_backbone=True  -> only the head is trained (fast, good default
                              for a few hundred images/class).
    freeze_backbone=False -> full fine-tuning (slower, use a low LR).
    unfreeze_last_n       -> optionally unfreeze only the last N backbone
                              layers for partial fine-tuning (a good middle
                              ground once the head has converged).
    """

    def __init__(
        self,
        num_classes: int,
        weights_path: str = "yolo11n.pt",
        feature_channels: int = BACKBONE_OUT_CHANNELS,
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = load_yolov11_backbone(weights_path)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_channels, num_classes)

        self._set_backbone_trainable(freeze_backbone, unfreeze_last_n)

    def _set_backbone_trainable(self, freeze_backbone: bool, unfreeze_last_n: int):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze_backbone

        if freeze_backbone and unfreeze_last_n > 0:
            for layer in list(self.backbone.children())[-unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True

    def forward(self, x):
        feats = self.backbone(x)          # (B, C, 7, 7)
        pooled = self.pool(feats)         # (B, C, 1, 1)
        pooled = torch.flatten(pooled, 1) # (B, C)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # quick sanity check
    model = FoodYOLOClassifier(num_classes=10, weights_path="yolo11n.pt", freeze_backbone=True)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("output shape:", out.shape)
    print("trainable params:", model.trainable_parameter_count())
    print("total params:", model.total_parameter_count())
