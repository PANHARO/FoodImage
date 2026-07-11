# FoodImage

Image-classification and YOLO preprocessing tools for a ten-class Southeast
Asian food dataset.

## Project layout

- `dataset/` - classification images split into `train`, `val`, and `test`
- `SEA_food_detection_cleaned_yolov11/` - YOLO images, labels, and `data.yaml`
- `trained_models/` - model scripts, shared graph utilities, and `checkpoints/`
- `reports/` - text reports and generated model statistics
- `logs/` - training logs
- `download_laphet_thoke.py` - builds the Laphet Thoke classification class
- `preprocess_yolo.py` - denoises, normalizes, and sharpens a YOLO export

## Model commands

```powershell
python trained_models/mobilenet.py --train
python trained_models/mobilenet.py --evaluate
python trained_models/efficientnet_setup.py --train
python trained_models/efficientnet_setup.py --evaluate
```

Training creates an epoch-loss/accuracy graph and CSV. Evaluation creates a
normalized confusion matrix plus per-class precision, recall, and F1 graphs.
Outputs are grouped under `reports/plots/mobilenet_v2/` and
`reports/plots/efficientnet_b0/`.

## YOLO preprocessing

```powershell
python preprocess_yolo.py `
  --input_dir SEA_food_detection_cleaned_yolov11 `
  --output_dir SEA_food_detection_preprocessed
```
