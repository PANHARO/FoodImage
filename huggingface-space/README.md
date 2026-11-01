---
title: SEA Food Recognition | Ro / Zenpai
emoji: 🍜
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
suggested_hardware: cpu-basic
pinned: false
short_description: Ten Southeast Asian dishes, recognized from a photo.
---

# SEA Food Recognition

A food-recognition portfolio demo by **Ro / Zenpai**, an IT and CS student in Phnom Penh. Upload a photo or try one of five examples for the top three EfficientNet-B0 predictions.

**Reported best validation accuracy: 81.7%.** The [training log](https://github.com/PANHARO/FoodImage/blob/11d56db391ac13343caaec9971bf8e7a6192e71f/logs/efficientnet.log) records 0.817. **Reported test accuracy: 82% on 415 images**, matching slides 18 and 32 of the [project presentation](https://www.canva.com/design/DAHOw_Ia42M/SAQgi9TKEnQeDsP9ovge1g/edit). These are historical project results, not a fresh evaluation of this Space. The log names a different checkpoint filename; no supplied metadata proves that its reported validation result belongs to the included weight bytes.

The source project combines dish classification, ingredient detection, and nutrition estimation. This Space demonstrates dish classification from that project.

## Known dishes

Adobo · Amok trey · Banh mi · Hainanese chicken rice · Laksa · Laphet thoke · Nasi goreng · Pad Thai · Pho · Satay

Below 50% top confidence, the app returns “Not sure” and the complete dish list, clearing the prediction bars. Scores are softmax probabilities over the ten known classes, not a guarantee of correctness or a reliable detector of unfamiliar food.

## Model and inference

- Uses the confirmed [inference code](https://github.com/aurora-llin/SEA-cuisine-Nutrition-estimation/blob/63722755091ce4779773df2323d601aea97b6a25/src/model_utils.py) and its `model/efficientnet-b0.pth` checkpoint.
- RGB → `Resize((224, 224))` → `ToTensor()` → normalization mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
- Preserves class order and original softmax/top-k inference; CPU, `eval()`, `torch.no_grad()`, loaded once at startup.
- The training repository's dataset-specific normalization is intentionally not used, following the owner's confirmed inference choice.
- Checkpoint SHA-256: `213f7d01da9716fef0e1268ff07b72f3ece4c14db09ae70a3adcad015edb48eb`.

## Run locally

Use Python 3.12 and run from this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open [localhost:7860](http://localhost:7860). The model and examples are bundled; inference needs no API key.

See [DEPLOY.md](DEPLOY.md) for Hugging Face upload and Git LFS instructions, and [embed.html](embed.html) for the portfolio iframe. Free CPU Spaces can sleep when idle; the embed includes a wake-up note. Photo sources are recorded in [examples/SOURCES.md](examples/SOURCES.md).

Project context: Group 3, ITM-390: Machine Learning, *Deep Learning-Based Southeast Asian Food Recognition and Nutritional Estimation*. The Canva source was read without changing its content.
