# FoodImage

A Python computer vision toolkit for training and evaluating MobileNetV2 and EfficientNet-B0 on ten Southeast Asian dishes, with preprocessing tools for a YOLO ingredient-detection dataset.

This repository contains training scripts and datasets, rather than a web application. The supported dish classes are adobo, amok trey, banh mi, Hainanese chicken rice, laksa, laphet thoke, nasi goreng, pad Thai, pho, and satay.

## Clone and install

Install Git and Python with a compatible PyTorch environment. Use a virtual environment and run all commands from the repository root.

```sh
git clone https://github.com/PANHARO/FoodImage.git
cd FoodImage
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your system uses `python3` instead of `python`, use it when creating the environment. Training downloads torchvision's pretrained ImageNet weights on first use. CUDA is used when available; CPU training is possible but slower.

## Train and evaluate

The classification dataset must have `dataset/train`, `dataset/val`, and `dataset/test`, each containing the same ten class folders. Keep the class names consistent with the model configuration files.

Choose a model and train it before evaluating the generated checkpoint:

```sh
python -m trained_models.mobilenet.cli --train
python -m trained_models.mobilenet.cli --evaluate
```

```sh
python -m trained_models.efficientnet.cli --train
python -m trained_models.efficientnet.cli --evaluate
```

Training produces epoch loss/accuracy plots and CSV files. Evaluation produces a normalized confusion matrix and per-class precision, recall, and F1 plots under `reports/plots/mobilenet_v2/` or `reports/plots/efficientnet_b0/`.

**Using the included checkpoints:** the configured evaluation paths currently expect `trained_models/checkpoints/best_mobilenet_v2_model.pth` and `trained_models/checkpoints/best_model.pth`. The included files are named `mobilenet_v2_model.pth` and `efficientnetb0_model.pth`. To evaluate those directly, update `MODEL_PATH` in the corresponding `trained_models/mobilenet/config.py` or `trained_models/efficientnet/config.py` to the matching included file. Otherwise, train first to generate the expected checkpoint.

The legacy training entry points also remain available:

```sh
python trained_models/mobilenet.py --train
python trained_models/efficientnet_setup.py --train
```

## Preprocess the YOLO dataset

```sh
python preprocess_yolo.py --input_dir SEA_food_detection_cleaned_yolov11 --output_dir SEA_food_detection_preprocessed
```

The script processes the YOLO `train`, `valid`, and `test` image splits using denoising, contrast enhancement, sharpening, and upscaling. It copies labels and `data.yaml` to the output. The default target size is 640; use `--target_size` to change it. Review the copied YAML paths before using the output with a YOLO trainer.

## Project layout

| Path | Purpose |
| --- | --- |
| `dataset/` | Classification images in train/val/test splits |
| `trained_models/mobilenet/` | MobileNetV2 configuration, training, and evaluation |
| `trained_models/efficientnet/` | EfficientNet-B0 configuration, training, and evaluation |
| `trained_models/checkpoints/` | Saved model weights |
| `SEA_food_detection_cleaned_yolov11/` | YOLO images, labels, and dataset configuration |
| `reports/`, `logs/` | Model reports, plots, and training logs |
| `preprocess_yolo.py` | YOLO dataset preprocessing |
| `download_image.py` | Laphet thoke image collection and dataset rebuilding |

`download_image.py` changes dataset contents, including rebuilding the laphet thoke class. Review the script and back up your dataset before running it. Model results depend on the dataset and training configuration; no accuracy guarantee is implied.

## Making changes

Fork the repository if you do not have write access, clone your fork, and create a branch with `git switch -c your-change`. Keep generated environments, private data, and unnecessary checkpoints out of commits. Describe any dataset or configuration changes when proposing a contribution.
