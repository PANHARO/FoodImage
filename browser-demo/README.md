# SEA Food Recognition — browser edition

A complete static browser app for Ro / Zenpai's portfolio. The exact confirmed EfficientNet-B0 checkpoint is exported to ONNX, with no quantization. ONNX Runtime Web runs CPU inference in a Web Worker. No uploaded photo is sent to a server.

## Run

Use a static HTTP server, for example `python -m http.server 8766 --directory dist`. Open localhost:8766. The model and WebAssembly runtime are bundled, so no API keys or external runtime CDN are required.

## Inference

Class order: adobo, amok_trey, banh_mi, hainanese_chicken_rice, laksa, laphet_thoke, nasi_goreng, pad_thai, pho, satay.

Preprocessing mirrors the confirmed [original inference](https://github.com/aurora-llin/SEA-cuisine-Nutrition-estimation/blob/63722755091ce4779773df2323d601aea97b6a25/src/model_utils.py): RGB, bilinear antialiased resize to 224 × 224, ToTensor equivalent, and normalization mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]. The separable 8-bit resize matches Pillow byte-for-byte on four checked image sizes. Browser image decoding and color handling can differ from Pillow for some files.

The original PyTorch model was loaded on CPU in eval mode; export and verification used torch.no_grad(). A single browser inference session is reused. The ONNX graph includes softmax over all ten classes. The UI shows the original top-three probabilities without renormalization. Below a top score of 0.5, it clears the bars and shows “Not sure” with all ten dishes.

Checkpoint SHA-256: 213f7d01da9716fef0e1268ff07b72f3ece4c14db09ae70a3adcad015edb48eb.

## Verification

- ONNX checker passed.
- Maximum PyTorch/ONNX probability difference over five examples: 0.00000011920928955078125.
- All five browser examples produce the same class and near-identical scores.
- Four raw RGB resize fixtures match Pillow exactly.
- Clearing removes prior predictions without reloading the model.
- Responsive layout checked at 641 px without horizontal overflow.
- The browser's actual image-upload and uncertain-result path was exercised.

See conversion-check.json for model-export results. Historical project accuracy is 81.7% best validation and 82% test on 415 images, per the linked training log and Canva presentation; this is not a new browser accuracy evaluation, and the log uses a different checkpoint filename.

## Sources and licenses

Model: [original project](https://github.com/aurora-llin/SEA-cuisine-Nutrition-estimation).
Photos: see dist/examples/SOURCES.md. Runtime: onnxruntime-web 1.31.1, with its MIT license in dist/vendor/LICENSE.
The project presentation was read without modifying its content.

The deployed Site is private to the owner by default. Public portfolio embedding requires making the Site public in its sharing settings.
