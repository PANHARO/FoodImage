"""Export the confirmed checkpoint and verify ONNX probabilities against PyTorch."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image

SITE = Path(__file__).resolve().parents[1]
ROOT = SITE.parent
SOURCE = ROOT / "huggingface-space"
WEIGHTS = SOURCE / "model" / "efficientnet-b0.pth"
assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == "213f7d01da9716fef0e1268ff07b72f3ece4c14db09ae70a3adcad015edb48eb"
torch.set_num_threads(2)
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
model.eval()
preprocess = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
wrapped = nn.Sequential(model, nn.Softmax(dim=1)).eval()
destination = SITE / "dist" / "model" / "food.onnx"
destination.parent.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    torch.onnx.export(wrapped, torch.zeros(1, 3, 224, 224), destination,
                      input_names=["image"], output_names=["probabilities"],
                      opset_version=17, dynamo=False, external_data=False)
onnx.checker.check_model(str(destination))
options = ort.SessionOptions()
options.intra_op_num_threads = 2
session = ort.InferenceSession(str(destination), options, providers=["CPUExecutionProvider"])
report = []
for name in ["adobo", "amok_trey", "banh_mi", "pho", "satay"]:
    with Image.open(SOURCE / "examples" / f"{name}.jpg") as opened:
        image = opened.convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        expected = wrapped(tensor).numpy()
    actual = session.run(None, {"image": tensor.numpy()})[0]
    error = float(np.max(np.abs(expected - actual)))
    assert error < 1e-5, (name, error)
    report.append({"name": name, "probabilities": expected[0].tolist(), "max_error": error})
(SITE / "conversion-check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"model_bytes": destination.stat().st_size, "max_probability_error": max(r["max_error"] for r in report)}, indent=2))

fixtures = ROOT / "work" / "resize-fixtures"
fixtures.mkdir(exist_ok=True)
manifest = []
for width, height in [(73, 101), (517, 307), (224, 224), (1, 37)]:
    data = bytes((i * 37 + i // 31) % 256 for i in range(width * height * 3))
    image = Image.frombytes("RGB", (width, height), data)
    key = f"{width}x{height}"
    (fixtures / f"{key}.rgb").write_bytes(data)
    (fixtures / f"{key}.expected").write_bytes(image.resize((224,224), Image.Resampling.BILINEAR).tobytes())
    manifest.append({"key": key, "width": width, "height": height})
(fixtures / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
