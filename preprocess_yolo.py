"""Preprocess a Roboflow YOLO export without changing its annotations."""

import argparse
import shutil
from pathlib import Path
import cv2


def upscale_and_sharpen(img, target_size=640):
    """Upscale small images proportionally, then apply mild sharpening."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)

    if scale > 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(img, 1.4, blurred, -0.4, 0)
    return sharpened


def denoise(img):
    """Remove color noise while preserving most image edges."""
    return cv2.fastNlMeansDenoisingColored(
        img, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21
    )


def normalize_exposure(img):
    """Improve local contrast using CLAHE on the luminance channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def process_image(img_path, output_path, target_size=640):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"    SKIPPED (couldn't read): {img_path.name}")
        return False

    img = denoise(img)
    img = normalize_exposure(img)
    img = upscale_and_sharpen(img, target_size=target_size)

    cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def process_split(split_name, input_dir, output_dir, target_size):
    img_in = input_dir / split_name / "images"
    lbl_in = input_dir / split_name / "labels"
    img_out = output_dir / split_name / "images"
    lbl_out = output_dir / split_name / "labels"

    if not img_in.exists():
        print(f"  ({split_name}: no images/ folder found, skipping)")
        return 0, 0

    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    images = [p for p in img_in.iterdir() if p.suffix.lower() in exts]

    success, missing_labels = 0, 0
    for img_path in images:
        out_img_path = img_out / img_path.name
        if process_image(img_path, out_img_path, target_size=target_size):
            success += 1

        # copy matching label file unchanged (same stem, .txt extension)
        label_path = lbl_in / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy2(label_path, lbl_out / label_path.name)
        else:
            missing_labels += 1

    return success, missing_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_size", type=int, default=640)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: input_dir not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    total_success, total_missing = 0, 0
    for split in ["train", "valid", "test"]:
        print(f"Processing {split}...")
        success, missing = process_split(split, input_dir, output_dir, args.target_size)
        print(f"  {success} images processed, {missing} missing label files")
        total_success += success
        total_missing += missing

    # Copy data.yaml over unchanged (paths inside it are relative, so no edits needed
    # as long as you keep the same train/valid/test/images/labels folder layout)
    yaml_src = input_dir / "data.yaml"
    if yaml_src.exists():
        shutil.copy2(yaml_src, output_dir / "data.yaml")
        print("\nCopied data.yaml")

    print(f"\nDone. {total_success} images processed total.")
    if total_missing:
        print(f"WARNING: {total_missing} images had no matching label file — check these before training.")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
