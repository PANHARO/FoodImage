"""
app.py -- SEA Food Recognition & Nutrition Estimator (Streamlit)

End-to-end demo per the ITM-390 proposal:
  1. Food Image Classification Module  -- YOLOv11 backbone + custom head
  2. Ingredient Management Module      -- editable per-dish ingredient list
  3. Nutrition Estimation Module       -- recalculates from final ingredients

Run from the project root:
    streamlit run src/app.py
"""

import os

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import FoodYOLOClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT_PATH = os.path.join(ROOT, "checkpoints", "best_model.pt")
INGREDIENTS_CSV = os.path.join(ROOT, "nutrition", "ingredients.csv")
DISH_MAP_CSV = os.path.join(ROOT, "nutrition", "dish_ingredients.csv")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DISPLAY_NAMES = {
    "adobo": "Adobo (Philippines)",
    "amok_trey": "Amok Trey (Cambodia)",
    "banh_mi": "Bánh Mì (Vietnam)",
    "hainanese_chicken_rice": "Hainanese Chicken Rice (Singapore/Malaysia)",
    "laksa": "Laksa (Malaysia/Singapore)",
    "laphet_thoke": "Laphet Thoke (Myanmar)",
    "nasi_goreng": "Nasi Goreng (Indonesia)",
    "pad_thai": "Pad Thai (Thailand)",
    "pho": "Phở (Vietnam)",
    "satay": "Satay (Indonesia/Malaysia)",
}


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    classes = ckpt["classes"]
    model = FoodYOLOClassifier(
        num_classes=len(classes), freeze_backbone=True, unfreeze_last_n=3
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, classes, device


@st.cache_data
def load_nutrition():
    ingredients = pd.read_csv(INGREDIENTS_CSV).set_index("ingredient")
    dish_map = pd.read_csv(DISH_MAP_CSV)
    return ingredients, dish_map


def classify(model, classes, device, pil_image, topk=3):
    # 4-view test-time augmentation (original, h-flip, center-zoom, zoom-flip)
    img = pil_image.convert("RGB")
    norm = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    base = img.resize((224, 224))
    w, h = img.size
    zoom = img.crop((int(w*0.05), int(h*0.05), int(w*0.95), int(h*0.95))).resize((224, 224))
    views = [base, base.transpose(Image.FLIP_LEFT_RIGHT),
             zoom, zoom.transpose(Image.FLIP_LEFT_RIGHT)]
    x = torch.stack([norm(v) for v in views]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).mean(dim=0)
    top = torch.topk(probs, k=min(topk, len(classes)))
    return [(classes[i], p.item()) for p, i in zip(top.values, top.indices)]


def compute_nutrition(rows, ingredients_db):
    """rows: list of dicts with 'ingredient' and 'quantity_g'."""
    totals = {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0}
    detail = []
    for r in rows:
        name, qty = r["ingredient"], float(r["quantity_g"])
        if name not in ingredients_db.index or qty <= 0:
            continue
        per100 = ingredients_db.loc[name]
        factor = qty / 100.0
        row = {
            "ingredient": name,
            "quantity_g": qty,
            "calories": per100["calories"] * factor,
            "protein": per100["protein"] * factor,
            "fat": per100["fat"] * factor,
            "carbs": per100["carbs"] * factor,
        }
        detail.append(row)
        for k in totals:
            totals[k] += row[k]
    return totals, pd.DataFrame(detail)


def main():
    st.set_page_config(page_title="SEA Food Recognition & Nutrition", page_icon="🍜", layout="wide")
    st.title("🍜 Southeast Asian Food Recognition & Nutrition Estimation")
    st.caption(
        "ITM-390 Machine Learning project · YOLOv11 backbone + custom classification head. "
        "Nutritional values are rough educational estimates, not medical or dietary advice."
    )

    model, classes, device = load_model()
    ingredients_db, dish_map = load_nutrition()

    uploaded = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png", "webp"])

    if uploaded is None:
        st.info("Upload a photo of one of the 10 supported dishes: "
                + ", ".join(DISPLAY_NAMES[c] for c in classes))
        return

    col_img, col_pred = st.columns([1, 1])
    image = Image.open(uploaded)
    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)

    predictions = classify(model, classes, device, image, topk=3)

    with col_pred:
        st.subheader("Model predictions")
        for name, prob in predictions:
            st.progress(prob, text=f"{DISPLAY_NAMES.get(name, name)} — {prob*100:.1f}%")

        top_class = predictions[0][0]
        # let the user override the prediction (handles misclassification gracefully)
        chosen = st.selectbox(
            "Dish (change if the prediction is wrong):",
            options=classes,
            index=classes.index(top_class),
            format_func=lambda c: DISPLAY_NAMES.get(c, c),
        )

    st.divider()
    st.subheader("2 · Ingredient Management")
    st.caption("Standard recipe loaded below — edit quantities, delete rows, or add ingredients to match your actual meal.")

    default_rows = dish_map[dish_map["dish"] == chosen][["ingredient", "quantity_g"]]

    # editable table; ingredient column restricted to known DB entries so
    # nutrition lookup never fails
    edited = st.data_editor(
        default_rows.reset_index(drop=True),
        num_rows="dynamic",
        column_config={
            "ingredient": st.column_config.SelectboxColumn(
                "ingredient", options=sorted(ingredients_db.index.tolist()), required=True
            ),
            "quantity_g": st.column_config.NumberColumn(
                "quantity (g)", min_value=0, max_value=2000, step=5
            ),
        },
        use_container_width=True,
        key=f"editor_{chosen}",
    )

    st.divider()
    st.subheader("3 · Nutrition Estimate")

    totals, detail = compute_nutrition(edited.to_dict("records"), ingredients_db)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Calories", f"{totals['calories']:.0f} kcal")
    m2.metric("Protein", f"{totals['protein']:.1f} g")
    m3.metric("Fat", f"{totals['fat']:.1f} g")
    m4.metric("Carbs", f"{totals['carbs']:.1f} g")

    with st.expander("Per-ingredient breakdown"):
        if not detail.empty:
            st.dataframe(
                detail.style.format({
                    "quantity_g": "{:.0f}",
                    "calories": "{:.0f}",
                    "protein": "{:.1f}",
                    "fat": "{:.1f}",
                    "carbs": "{:.1f}",
                }),
                use_container_width=True,
            )

    st.caption(
        "Estimates use per-100g reference values (USDA FoodData Central / ASEAN Food "
        "Composition style) and typical single-serving recipes. Actual dishes vary widely."
    )


if __name__ == "__main__":
    main()
