"""
robustness_test.py

Self-generated robustness benchmark: applies controlled corruptions to the
real held-out test set and measures accuracy degradation per corruption.
Simulates real-world phone-photo conditions (motion blur, low light, noise,
partial occlusion, heavy JPEG compression, off-angle tilt).

Usage:
    python robustness_test.py --data ../dataset --ckpt ../checkpoints/best_model.pt
"""

import argparse
import io
import json
import os
import random

import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from torchvision import datasets, transforms

from model import FoodYOLOClassifier

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
random.seed(123)


def corrupt_blur(img):        return img.filter(ImageFilter.GaussianBlur(radius=2.5))
def corrupt_lowlight(img):    return ImageEnhance.Brightness(img).enhance(0.45)
def corrupt_overexposed(img): return ImageEnhance.Brightness(img).enhance(1.7)

def corrupt_noise(img):
    import numpy as np
    a = np.array(img).astype(np.int16)
    a = a + np.random.normal(0, 22, a.shape)
    return Image.fromarray(a.clip(0, 255).astype('uint8'))

def corrupt_occlusion(img):
    """Random opaque block covering ~15% of the image (hand/utensil over dish)."""
    img = img.copy()
    w, h = img.size
    bw, bh = int(w * 0.38), int(h * 0.38)
    x, y = random.randint(0, w - bw), random.randint(0, h - bh)
    ImageDraw.Draw(img).rectangle([x, y, x + bw, y + bh], fill=(120, 110, 100))
    return img

def corrupt_jpeg(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=12)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def corrupt_tilt(img):
    return img.rotate(28, expand=False, fillcolor=(255, 255, 255))


CORRUPTIONS = {
    "clean (baseline)": None,
    "gaussian_blur": corrupt_blur,
    "low_light": corrupt_lowlight,
    "overexposed": corrupt_overexposed,
    "sensor_noise": corrupt_noise,
    "occlusion_38pct": corrupt_occlusion,
    "jpeg_q12": corrupt_jpeg,
    "tilt_28deg": corrupt_tilt,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../dataset")
    p.add_argument("--ckpt", default="../checkpoints/best_model.pt")
    p.add_argument("--out", default="../results/robustness.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    classes = ck["classes"]
    model = FoodYOLOClassifier(num_classes=len(classes), freeze_backbone=True, unfreeze_last_n=3)
    model.load_state_dict(ck["model_state_dict"]); model.to(device).eval()

    base_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_ds = datasets.ImageFolder(os.path.join(args.data, "test"))
    assert test_ds.classes == classes

    results = {}
    for name, fn in CORRUPTIONS.items():
        correct = total = 0
        batch, labels = [], []
        with torch.no_grad():
            for path, y in test_ds.samples:
                img = Image.open(path).convert("RGB")
                if fn is not None:
                    img = fn(img)
                batch.append(base_tf(img)); labels.append(y)
                if len(batch) == 64 or (path, y) == test_ds.samples[-1]:
                    x = torch.stack(batch).to(device)
                    preds = model(x).argmax(1).cpu()
                    correct += (preds == torch.tensor(labels)).sum().item()
                    total += len(labels)
                    batch, labels = [], []
        acc = correct / total
        results[name] = acc
        print(f"{name:22s} accuracy = {acc*100:5.1f}%")

    baseline = results["clean (baseline)"]
    print("\nDegradation vs clean:")
    for name, acc in results.items():
        if name != "clean (baseline)":
            print(f"  {name:22s} {(acc-baseline)*100:+5.1f} pts")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
