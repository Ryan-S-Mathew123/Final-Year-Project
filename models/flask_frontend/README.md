# Bird Model Comparative Dashboard

A Flask application that compares the same four classifier families for audio, images, and video: Random Forest, SVM, K-Nearest Neighbors, and Gradient Boosting. Video analysis applies the image classifiers to sampled frames and the audio classifiers to the extracted sound track.

## Features

- Audio classification from 40 MFCC features.
- Image classification from features extracted by the existing ConvNeXt checkpoint.
- The same four regularized classifier families for both modalities.
- Held-out test accuracy and macro F1 alongside each model's per-upload prediction.
- Per-model predicted species, confidence, top-3 probabilities, and inference latency.
- Majority-vote consensus summary.
- Audio playback and image preview before analysis.
- Models are loaded once when the Flask application starts.

## Setup

Run these commands from the repository root:

```powershell
python -m pip install -r models/flask_frontend/requirements.txt
```

Train the shared model families from the repository root:

```powershell
python models/audio_model/train.py
python models/image_model/train_classifiers.py
```

The image training command requires the ConvNeXt checkpoint at `models/image_model/checkpoints/convnext_best.pth`. Each training command uses fixed train/validation/test data, saves model artifacts, and writes metrics to its modality's `results/classifier_comparison.json`.

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
- `latency_ms`: classifier inference time
- `top_three`: top-3 class probabilities
- `test_accuracy` and `test_macro_f1`: scores on the held-out test split
- `consensus`: majority-vote summary

## Model files

The audio and image model files are saved under:

```text
models/{audio_model,image_model}/comparison_models/{random_forest,svm,knn,gradient_boosting}.joblib
```

The image feature extractor uses `models/image_model/checkpoints/convnext_best.pth`. Missing or unreadable artifacts are reported by `/health` and on the dashboard page.

## Preprocessing

Audio files are loaded as mono audio and converted into 40 averaged MFCC features.

Images are converted to RGB, resized to 256 pixels, center-cropped to 224 x 224, normalized with ImageNet mean and standard deviation values, and encoded by ConvNeXt before classification. The reported classifier latency does not include feature extraction.

The audio training script creates a deterministic, stratified 70/15/15 train/validation/test split. Image training uses the existing train/val/test folders. Test scores are reported for comparison and are not used to fit the classifiers.

## Project files

- `app.py`: Flask server, model loading, preprocessing, inference, and API routes.
- `templates/index.html`: Dashboard markup.
- `static/css/styles.css`: Dashboard styling.
- `static/js/main.js`: Upload handling, AJAX requests, and result rendering.
- `requirements.txt`: Python dependencies.
