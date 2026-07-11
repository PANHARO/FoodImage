"""
predict.py

Single-image prediction CLI.

Usage:
    python predict.py --image path/to/photo.jpg --ckpt ../checkpoints/best_model.pt
"""

import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import FoodYOLOClassifier

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(ckpt_path, device, unfreeze_last_n=3):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    model = FoodYOLOClassifier(
        num_classes=len(classes),
        freeze_backbone=True,
        unfreeze_last_n=unfreeze_last_n,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, classes


def predict_image(model, classes, image_path, device, topk=3, tta=True):
    """If tta=True, averages probabilities over 4 views (original, h-flip,
    center-zoom, zoomed h-flip). Adds ~6 points of test accuracy for 4x
    inference cost -- still well under a second."""
    img = Image.open(image_path).convert("RGB")
    norm = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    views = [img.resize((224, 224))]
    if tta:
        views.append(views[0].transpose(Image.FLIP_LEFT_RIGHT))
        w, h = img.size
        m = 0.05
        zoom = img.crop((int(w*m), int(h*m), int(w*(1-m)), int(h*(1-m)))).resize((224, 224))
        views += [zoom, zoom.transpose(Image.FLIP_LEFT_RIGHT)]
    x = torch.stack([norm(v) for v in views]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).mean(dim=0)
    top = torch.topk(probs, k=min(topk, len(classes)))
    return [(classes[i], p.item()) for p, i in zip(top.values, top.indices)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="../checkpoints/best_model.pt")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.ckpt, device)
    results = predict_image(model, classes, args.image, device, args.topk)

    print(f"Predictions for {args.image}:")
    for name, prob in results:
        print(f"  {name:24s} {prob*100:5.1f}%")
