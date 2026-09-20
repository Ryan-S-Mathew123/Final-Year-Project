# Bird Model Comparative Dashboard

A Flask web application for comparing the four trained audio classifiers and four trained image classifiers side by side.

## Features

- Audio classification with Random Forest, Gradient Boosting, SVM, and K-Nearest Neighbors.
- Image classification with ResNet50, EfficientNet-B0, MobileNetV3-Large, and ConvNeXt-Tiny.
- Per-model predicted species, confidence, top-3 probabilities, and inference latency.
- Majority-vote consensus summary.
- Audio playback and image preview before analysis.
- Models are loaded once when the Flask application starts.

## Setup

Run these commands from the repository root:

```powershell
python -m pip install -r models/flask_frontend/requirements.txt
```

## Start the dashboard

```powershell
python models/flask_frontend/app.py
```

Open the dashboard at:

```text
http://127.0.0.1:5000
```

## API endpoints

### `GET /health`

Returns the loaded audio and image models, device information, and model-loading errors.

### `POST /predict/audio`

Accepts a multipart upload using the field name `file`.

Supported formats:

- WAV
- MP3
- OGG
- FLAC
- M4A

Example PowerShell request:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict/audio -F "file=@sample.wav"
```

### `POST /predict/image`

Accepts a multipart upload using the field name `file`.

Supported formats:

- JPG/JPEG
- PNG
- WEBP

Example PowerShell request:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict/image -F "file=@bird.jpg"
```

Each prediction response contains:

- `results`: one result per loaded model
- `species`: predicted class
- `confidence`: confidence percentage
- `latency_ms`: inference time
- `top_three`: top-3 class probabilities
- `consensus`: majority-vote summary

## Model files

The application expects these files:

```text
models/audio_model/model.pkl
models/audio_model/model_gradient_boosting.pkl
models/audio_model/model_svm.pkl
models/audio_model/model_knn.pkl
models/audio_model/label_encoder.pkl
```

The image checkpoints are expected at:

```text
models/image_model/checkpoints/resnet50_best.pth
models/image_model/checkpoints/efficientnet_best.pth
models/image_model/checkpoints/mobilenet_best.pth
models/image_model/checkpoints/convnext_best.pth
```

If a model cannot be loaded, the dashboard remains available and the problem is listed by `/health` and on the dashboard page.

## Preprocessing

Audio files are loaded as mono audio and converted into 40 averaged MFCC features, matching the audio training pipeline.

Images are converted to RGB, resized to 256 pixels, center-cropped to 224 x 224, converted to tensors, and normalized with ImageNet mean and standard deviation values.

## Project files

- `app.py`: Flask server, model loading, preprocessing, inference, and API routes.
- `templates/index.html`: Dashboard markup.
- `static/css/styles.css`: Dashboard styling.
- `static/js/main.js`: Upload handling, AJAX requests, and result rendering.
- `requirements.txt`: Python dependencies.
