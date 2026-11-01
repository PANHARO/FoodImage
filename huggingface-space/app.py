"""Ro / Zenpai's Southeast Asian food-recognition demo."""
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "model" / "efficientnet-b0.pth"
CONFIDENCE_THRESHOLD = 0.5

# Keep this order identical to the trained classifier's output indices.
CLASS_NAMES = [
    "adobo", "amok_trey", "banh_mi", "hainanese_chicken_rice", "laksa",
    "laphet_thoke", "nasi_goreng", "pad_thai", "pho", "satay",
]
DISPLAY_NAMES = [
    "Adobo", "Amok trey", "Banh mi", "Hainanese chicken rice", "Laksa",
    "Laphet thoke", "Nasi goreng", "Pad Thai", "Pho", "Satay",
]
DISH_NAMES = dict(zip(CLASS_NAMES, DISPLAY_NAMES))

# Copied exactly from the authoritative src/model_utils.py.
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    """Load once on CPU. No ImageNet download or GPU is needed."""
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {WEIGHTS_PATH}. Copy the trained weights into model/."
        )
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(CLASS_NAMES)
    )
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to("cpu")
    model.eval()
    return model


# This runs at startup, never inside a prediction request.
# Two threads suit the free CPU Space and avoid excessive thread overhead.
torch.set_num_threads(2)
MODEL = load_model()


def predict_dish(model, image: Image.Image, top_k=3):
    """Original inference function, including no_grad and full softmax."""
    x = _preprocess(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top_probs, top_idx = torch.topk(probs, top_k)
    return [(CLASS_NAMES[i], float(p)) for p, i in zip(top_probs, top_idx)]


def recognize_food(image: Image.Image | None):
    """Convert uploads to RGB and abstain when the top score is below 50%."""
    if image is None:
        return "Upload a food photo or choose an example below.", None

    predictions = predict_dish(MODEL, image.convert("RGB"), top_k=3)
    best_class, confidence = predictions[0]
    if confidence < CONFIDENCE_THRESHOLD:
        # Clear the label as well, so a previous confident result cannot remain.
        return (
            "Not sure. This model only knows these 10 dishes: "
            + ", ".join(DISPLAY_NAMES),
            None,
        )

    # Keep probabilities from all 10 classes; do not renormalize the top three.
    scores = {DISH_NAMES[name]: score for name, score in predictions}
    return f"**{DISH_NAMES[best_class]}** · {confidence:.1%} confidence", scores


CSS = """
html, body, .gradio-container { background: #080808 !important; }
.gradio-container {
    max-width: 1100px !important; margin: 0 auto;
    padding: 32px 24px !important; color: #eff5f1;
}
.gradio-container h1 {
    font-size: clamp(30px, 5vw, 48px) !important;
    font-weight: 600 !important; letter-spacing: -1.8px;
    line-height: 1.1 !important;
}
.gradio-container .block:has(h1) { overflow: visible !important; }
.gradio-container a { color: #59e0a0 !important; }
.gradio-container .prose { color: #aab5ae; }
.gradio-container .prose strong { color: #eff5f1; }
#result { min-height: 62px; padding: 14px 16px; }
#result strong { color: #59e0a0; font-size: 22px; }
#predictions { --stat-background-fill: #59e0a0 !important; }
#predictions .confidence-set { color: #eff5f1; }
.gradio-container button:focus-visible, .gradio-container a:focus-visible {
    outline: 2px solid #59e0a0 !important; outline-offset: 3px;
}
@media (max-width: 600px) {
    .gradio-container { padding: 24px 14px !important; }
}
"""

THEME = gr.themes.Base(
    primary_hue="green", neutral_hue="zinc", font=["system-ui", "sans-serif"],
).set(
    body_background_fill="#080808", body_background_fill_dark="#080808",
    body_text_color="#eff5f1", body_text_color_dark="#eff5f1",
    body_text_color_subdued="#aab5ae", body_text_color_subdued_dark="#aab5ae",
    block_background_fill="#111411", block_background_fill_dark="#111411",
    block_border_color="#29332c", block_border_color_dark="#29332c",
    input_background_fill="#0c100d", input_background_fill_dark="#0c100d",
    button_primary_background_fill="#59e0a0",
    button_primary_background_fill_dark="#59e0a0",
    button_primary_text_color="#06150d", button_primary_text_color_dark="#06150d",
    slider_color="#59e0a0", slider_color_dark="#59e0a0",
)

EXAMPLE_NAMES = ["adobo", "amok_trey", "banh_mi", "pho", "satay"]
demo = gr.Interface(
    fn=recognize_food,
    inputs=gr.Image(
        type="pil", image_mode="RGB", sources=["upload"],
        label="Your food photo", height=300,
    ),
    outputs=[
        gr.Markdown(
            value="Upload a food photo or choose an example below.", elem_id="result",
        ),
        gr.Label(
            num_top_classes=3, label="Top 3 predictions",
            show_heading=False, color="#59e0a0", elem_id="predictions",
        ),
    ],
    title="A taste of Southeast Asia.",
    description=(
        "**RO / ZENPAI · MACHINE LEARNING**\n\n"
        "Ten dishes. One photo. Upload a meal and explore what EfficientNet-B0 sees. "
        "Made by an IT & CS student in Phnom Penh."
    ),
    article=(
        "**Inside the project** · This classifier is part of Group 3's "
        "*Deep Learning-Based Southeast Asian Food Recognition and Nutritional "
        "Estimation* project for ITM-390: Machine Learning.\n\n"
        "**Known dishes:** " + ", ".join(DISPLAY_NAMES) + ".\n\n"
        "Scores below 50% return “Not sure.” A high score can still be wrong, "
        "especially for unfamiliar food.\n\n"
        "[Project source](https://github.com/aurora-llin/SEA-cuisine-Nutrition-estimation)"
        " · [Dataset & training](https://github.com/PANHARO/FoodImage)"
    ),
    examples=[[str(ROOT / "examples" / f"{name}.jpg")] for name in EXAMPLE_NAMES],
    examples_per_page=5,
    cache_examples=False,
    preload_example=False,
    live=True,  # Predict automatically when an upload or example changes.
    flagging_mode="never",
    analytics_enabled=False,
    api_name="predict",
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=20).launch(
        theme=THEME,
        css=CSS,
        js="() => { document.documentElement.classList.add('dark'); }",
    )
