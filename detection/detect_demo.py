"""
detect_demo.py -- run trained ingredient detector on an image and print/draw
boxes. Also shows how to bridge detections into the nutrition module.

Usage:
    python detect_demo.py --image photo.jpg --weights runs/ingredients/weights/best.pt
"""
import argparse
from ultralytics import YOLO

# rough per-instance gram estimates to seed the nutrition table
# (user still adjusts quantities in the app -- detection just pre-fills)
GRAMS_PER_INSTANCE = {
    "shrimp": 12, "chicken_piece": 25, "beef_slice": 15, "egg": 50,
    "noodles": 180, "rice": 200, "bean_sprouts": 40, "lime_wedge": 15,
    "peanuts": 15, "cucumber_slice": 8, "tomato": 30, "chili": 4, "herbs": 5,
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--weights", default="runs/ingredients/weights/best.pt")
    p.add_argument("--conf", type=float, default=0.35)
    args = p.parse_args()

    model = YOLO(args.weights)
    r = model.predict(args.image, conf=args.conf, verbose=False)[0]
    r.save("detected.jpg")  # image with boxes drawn

    counts = {}
    for b in r.boxes:
        name = model.names[int(b.cls)]
        counts[name] = counts.get(name, 0) + 1

    print(f"{len(r.boxes)} ingredient detections -> detected.jpg")
    print("\nPre-filled nutrition table rows:")
    for name, n in sorted(counts.items()):
        grams = n * GRAMS_PER_INSTANCE.get(name, 20)
        print(f"  {name:16s} x{n:<2d} -> ~{grams} g")

if __name__ == "__main__":
    main()
