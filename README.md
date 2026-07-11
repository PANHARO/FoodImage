# SEA Food Recognition & Nutrition Estimation
### ITM-390 Machine Learning — YOLOv11 backbone + custom PyTorch classification head

Recognizes 10 Southeast Asian dishes from photos and estimates nutrition
(calories, protein, fat, carbs) from an editable ingredient list.

Dishes: adobo, amok trey, bánh mì, hainanese chicken rice, laksa, laphet thoke,
nasi goreng, pad thai, phở, satay.

> **v2 note:** mohinga was replaced with laphet thoke (100 unique source
> photos, source-level split, 4x offline augmentation for training).
> v3 test results (after +17 new laphet photos): 74.8% accuracy single-view.
> **With 4-view test-time augmentation (now the default in predict.py and
> app.py): 81.2% accuracy, 0.79 macro F1** (393 images). Laphet thoke:
> precision 0.64, recall 0.50 — still data-limited — it needs more real
> photos (~300 sources) to match the other classes. The Colab/GPU
> train_best.py path applies unchanged to this dataset.

---

## Why this architecture (important for the report)

The professor required YOLOv11 but also required writing our own Python
rather than depending on YOLO end-to-end. The task is **single-label image
classification** (one dish per photo, no localization), so YOLO's detection
head — and mAP as a metric — are the wrong fit. Instead:

- We load pretrained **YOLOv11n** weights and keep only the **backbone**
  (layers 0–10: conv stem, C3k2 blocks, SPPF, C2PSA attention — the part
  that learns general visual features from COCO).
- The detection neck/head (layers 11–23) is **discarded**.
- We attach our **own classification head** (global average pool → dropout
  → linear layer) and write the **training loop, data pipeline, and
  evaluation entirely in plain PyTorch** — no `model.train()`, no
  ultralytics Trainer, no yolo11-cls task.

This is standard transfer learning, identical in spirit to the
EfficientNet-B0 plan in the proposal, just with a YOLOv11 backbone.

## Results (trained in this repo, CPU-only)

Two-phase training on the 4,186-image dataset (train/val/test = 80/10/10):

| Phase | What trains | LR | Val acc |
|---|---|---|---|
| 1. Linear probe (epochs 1–15) | head only (2,570 params) | 1e-3 | 37.6% (plateau) |
| 2. Fine-tune (epochs 16–31) | head + last 3 backbone layers (763K params) | 1e-4 | **78.0%** |

**Held-out test set (423 images): 74.0% accuracy · 0.75 macro precision ·
0.74 macro recall · 0.74 macro F1.**

The phase-1 plateau is itself a useful finding for the report: frozen
COCO-detection features alone can't separate fine-grained food classes;
fine-tuning the deeper backbone layers doubled accuracy. The confusion
matrix (results/confusion_matrix.png) shows the remaining errors
concentrate among the three visually-similar noodle soups
(phở / laksa / mohinga) — exactly what you'd expect.

### Getting above 80–85% (do this on GPU)

Training here was CPU-limited. On Colab (free GPU), the same code should
reach the proposal's 85% target with:

```bash
# full fine-tune: unfreeze everything, low LR, more epochs
python train.py --data ../dataset --epochs 60 --batch_size 64 \
    --lr 5e-5 --unfreeze_last_n 11 --patience 10 --resume
```

Other easy wins: swap to the larger `yolo11s.pt` backbone (change
`--weights` and pass `feature_channels=512` in model.py — see the
FEATURE_CHANNELS table), stronger augmentation (RandAugment), and label
smoothing.

## Project layout

```
food_yolo_classifier/
├── src/
│   ├── model.py        # backbone extraction + FoodYOLOClassifier
│   ├── dataset.py      # transforms + dataloaders
│   ├── train.py        # manual training loop (resume + early stopping)
│   ├── evaluate.py     # test metrics, confusion matrix, training curves
│   ├── predict.py      # single-image CLI
│   └── app.py          # Streamlit web app (3 modules per proposal)
├── nutrition/
│   ├── ingredients.csv        # 83 ingredients, per-100g macros
│   └── dish_ingredients.csv   # standard recipe per dish (grams)
├── checkpoints/
│   ├── best_model.pt   # trained weights (74% test acc)
│   ├── history.json    # full epoch-by-epoch log
│   └── classes.json
├── results/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── test_metrics.json
└── requirements.txt
```

## Setup & usage

```bash
pip install -r requirements.txt

# place your dataset folder (train/ val/ test/ with class subfolders)
# at the project root as ./dataset

cd src

# train (resumable — safe to interrupt and rerun with --resume)
python train.py --data ../dataset --epochs 20 --lr 1e-3            # phase 1
python train.py --data ../dataset --epochs 40 --lr 1e-4 \
    --unfreeze_last_n 3 --resume                                    # phase 2

# evaluate on test set
python evaluate.py --data ../dataset --ckpt ../checkpoints/best_model.pt

# predict one image
python predict.py --image /path/to/photo.jpg

# run the web app (from project root)
cd .. && streamlit run src/app.py
```

## The three modules (matches the proposal)

1. **Food Image Classification** — upload photo → top-3 predictions with
   confidence. User can override a wrong prediction from a dropdown.
2. **Ingredient Management** — the predicted dish loads its standard
   recipe; the user can add/remove/edit ingredients and quantities in an
   editable table (restricted to the 83-ingredient DB so lookups never fail).
3. **Nutrition Estimation** — totals recompute live from the final
   ingredient list using per-100g values (USDA / ASEAN-style references).

## Adding a new dish (e.g. laphet thoke)

1. Add `dataset/train/laphet_thoke/`, `dataset/val/...`, `dataset/test/...`
   with images (the 82 augmented images you have are a start; aim for
   300+ originals before augmentation).
2. Add its recipe rows to `nutrition/dish_ingredients.csv` (and any new
   ingredients to `ingredients.csv`).
3. Retrain — the class count is read from the folders automatically.

## Disclaimer

Educational estimates only. Per the proposal: not for medical, dietary, or
professional nutritional assessment.
