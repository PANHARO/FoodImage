"""Configuration for MobileNetV2 food classification."""

from pathlib import Path


# Dataset layout and class order must match torchvision.datasets.ImageFolder.
DATA_ROOT = Path("dataset")
MODEL_PATH = Path("trained_models/checkpoints/best_mobilenet_v2_model.pth")
MODEL_NAME = "mobilenet_v2"
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
