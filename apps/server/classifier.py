from pathlib import Path
import threading

import joblib
import librosa
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = ROOT / "models" / "audio_model"
IMAGE_DIR = ROOT / "models" / "image_model"
IMAGE_CHECKPOINTS = {
    "resnet50": IMAGE_DIR / "checkpoints" / "resnet50_best.pth",
    "efficientnet": IMAGE_DIR / "checkpoints" / "efficientnet_best.pth",
    "mobilenet": IMAGE_DIR / "checkpoints" / "mobilenet_best.pth",
    "convnext": IMAGE_DIR / "checkpoints" / "convnext_best.pth",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _create_image_model(name, classes):
    if name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, classes)
    elif name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, classes)
    elif name == "mobilenet":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, classes)
    elif name == "convnext":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, classes)
    else:
        raise ValueError(f"Unknown image model: {name}")
    return model


class ModelClassifier:
    """Loads the trained audio model and image checkpoints once per process."""

    def __init__(self):
        self._lock = threading.Lock()
        self._audio_model = None
        self._audio_labels = None
        self._image_models = {}

    def _load_audio(self):
        if self._audio_model is None:
            with self._lock:
                if self._audio_model is None:
                    self._audio_model = joblib.load(AUDIO_DIR / "model.pkl")
                    self._audio_labels = joblib.load(AUDIO_DIR / "label_encoder.pkl")
        return self._audio_model, self._audio_labels

    def classify_audio(self, path):
        model, labels = self._load_audio()
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
        features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        features = np.mean(features.T, axis=0).reshape(1, -1)
        probabilities = model.predict_proba(features)[0]
        best = int(np.argmax(probabilities))
        return {
            "species": str(labels.inverse_transform([best])[0]),
            "confidence": float(probabilities[best]),
            "model": "audio_random_forest_mfcc",
        }

    def _load_image(self, name):
        if name not in self._image_models:
            with self._lock:
                if name not in self._image_models:
                    checkpoint_path = IMAGE_CHECKPOINTS[name]
                    if not checkpoint_path.exists():
                        raise FileNotFoundError(f"Image checkpoint not found: {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                    classes = checkpoint["classes"]
                    model = _create_image_model(name, len(classes))
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.to(DEVICE).eval()
                    self._image_models[name] = (model, classes)
        return self._image_models[name]

    def classify_image(self, path):
        image = Image.open(path).convert("RGB")
        tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        predictions = {}
        with torch.inference_mode():
            for name in IMAGE_CHECKPOINTS:
                model, classes = self._load_image(name)
                probabilities = torch.softmax(model(tensor), dim=1)[0]
                best = int(torch.argmax(probabilities))
                predictions[name] = {
                    "species": str(classes[best]),
                    "confidence": float(probabilities[best].item()),
                }
        votes = {}
        for prediction in predictions.values():
            votes[prediction["species"]] = votes.get(prediction["species"], 0) + 1
        ensemble_species = max(votes, key=votes.get)
        ensemble_confidence = float(np.mean([
            p["confidence"] for p in predictions.values()
            if p["species"] == ensemble_species
        ]))
        return {
            "species": ensemble_species,
            "confidence": ensemble_confidence,
            "models": predictions,
            "device": str(DEVICE),
        }

class BirdClassifier:
    def __init__(self):
        print("Bird classifier placeholder initialized")

    def predict(self, audio, sample_rate=44100):
        # TODO: Replace this method with your trained bird classifier.
        if audio is None or len(audio)==0:
            return {"species":"No audio","confidence":0.0,"detected":False}
        rms=float(np.sqrt(np.mean(np.square(audio))))
        if rms>0.01:
            return {"species":"Bird detected (placeholder)","confidence":0.85,"detected":True}
        return {"species":"No bird detected","confidence":0.0,"detected":False}
