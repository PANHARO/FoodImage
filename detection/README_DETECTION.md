# Ingredient Detection Module (add-on)

Upgrades the pipeline from "classify dish -> load standard recipe" to
"detect actual ingredients -> pre-fill the nutrition table with what's
really visible in THIS photo."

## The one thing code cannot do: annotation

Detection training needs bounding boxes drawn by humans. Proof of why
pretrained YOLO can't shortcut this: run stock yolo11n on a pad thai photo
and it detects "bowl, wine glass, apple, broccoli" -- COCO has no shrimp,
noodles, or bean sprouts classes.

## Annotation plan (realistic scope)

- Tool: Roboflow (free tier, in-browser, exports YOLO format directly).
  Alternatives: CVAT, LabelImg.
- Scope: 300-400 images from the existing dataset, prioritizing dishes with
  countable ingredients (pad thai, laksa, pho, satay, nasi goreng).
- 13 visually-detectable classes (see data.yaml). Do NOT try to label
  sauces/oils -- they are invisible and stay recipe-based.
- Effort estimate: ~6 boxes/image average, ~8 s/box in Roboflow
  => 4-6 team-hours total. Split 4 ways = manageable in a weekend.
- Aim for >=100 instances per class; check Roboflow's class-balance chart.

## Labeling rules (agree on these BEFORE starting, consistency >> volume)

1. Box each VISIBLE instance separately (each shrimp its own box).
2. Mass items (rice, noodles, sprouts) get ONE box around the whole region.
3. Occluded >70% -> skip. Ambiguous -> skip (bad labels hurt more than
   missing ones).
4. Boxes tight to the ingredient, not the plate.

## Train (GPU) and integrate

    cd detection
    python train_detect.py          # ~1-2 h on the 5060 Ti
    python detect_demo.py --image test.jpg

Integration into the app: classification module stays untouched (it names
the dish and gives country/context); the detector's per-ingredient counts
pre-fill the editable nutrition table via GRAMS_PER_INSTANCE, replacing the
static standard recipe when boxes are found. The user still edits final
quantities -- detection makes the starting point smarter, not perfect.

## Metrics

This module correctly reports mAP50 / mAP50-95 (it IS detection).
The classifier continues reporting accuracy/precision/recall/F1.
Presenting both, each with its proper metric, directly answers the
professor's original YOLOv11 requirement AND explains the earlier <50% mAP:
that number came from forcing classification data into detection format
with auto-generated whole-image boxes.
