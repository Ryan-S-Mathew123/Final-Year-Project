"""Shared model loading and inference for the dashboard and API server."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
from PIL import Image

from models.image_model.feature_extractor import IMAGE_TRANSFORM, load_image_encoder
from models.model_families import MODEL_ARTIFACTS
from models.species import normalize_species_id


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "models" / "audio_model"
IMAGE_DIR = REPO_ROOT / "models" / "image_model"
AUDIO_SAMPLE_RATE = 22_050
AUDIO_FEATURE_COUNT = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_audio_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply the training-time MFCC preprocessing to file or live audio."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("Audio contains no samples.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains non-finite samples.")
    if sample_rate != AUDIO_SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=AUDIO_SAMPLE_RATE,
        )
    mfcc = librosa.feature.mfcc(y=audio, sr=AUDIO_SAMPLE_RATE, n_mfcc=AUDIO_FEATURE_COUNT)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)


class ModelRegistry:
    """Loads and serves the same classifier families for both modalities."""

    def __init__(self) -> None:
        self.audio_models: dict[str, tuple[object, list[str]]] = {}
        self.image_models: dict[str, tuple[object, list[str]]] = {}
        self.image_encoder: torch.nn.Module | None = None
        self.audio_metrics: dict = {}
        self.image_metrics: dict = {}
        self.load_errors: list[str] = []
        self._load_audio_models()
        self._load_image_models()

    def _load_metrics(self, directory: Path, modality: str) -> dict:
        metrics_path = directory / "results" / "classifier_comparison.json"
        try:
            with metrics_path.open(encoding="utf-8") as file:
                report = json.load(file)
            if not isinstance(report, dict) or report.get("modality") != modality:
                raise ValueError(f"Expected {modality} metrics in {metrics_path}")
            return report
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.load_errors.append(f"{modality.title()} evaluation metrics: {exc}")
            return {}

    def _load_audio_models(self) -> None:
        self.audio_metrics = self._load_metrics(AUDIO_DIR, "audio")
        for name, filename in MODEL_ARTIFACTS.items():
            path = AUDIO_DIR / "comparison_models" / filename
            try:
                artifact = joblib.load(path)
                model = artifact["model"]
                labels = [str(label) for label in artifact["classes"]]
                if not hasattr(model, "predict_proba"):
                    raise TypeError("Classifier does not implement predict_proba")
                self.audio_models[name] = (model, labels)
            except Exception as exc:
                self.load_errors.append(f"{name}: {exc}")

    def _load_image_models(self) -> None:
        self.image_metrics = self._load_metrics(IMAGE_DIR, "image")
        encoder_path = IMAGE_DIR / "checkpoints" / "convnext_best.pth"
        try:
            self.image_encoder, image_classes = load_image_encoder(encoder_path, DEVICE)
        except Exception as exc:
            self.load_errors.append(f"ConvNeXt image feature extractor: {exc}")
            image_classes = []

        for name, filename in MODEL_ARTIFACTS.items():
            path = IMAGE_DIR / "comparison_models" / filename
            try:
                artifact = joblib.load(path)
                model = artifact["model"]
                classes = [str(label) for label in artifact["classes"]]
                if not hasattr(model, "predict_proba"):
                    raise TypeError("Classifier does not implement predict_proba")
                if image_classes and classes != image_classes:
                    raise ValueError("Classifier labels do not match the ConvNeXt checkpoint.")
                self.image_models[name] = (model, classes)
            except Exception as exc:
                self.load_errors.append(f"{name}: {exc}")

    @staticmethod
    def _top_three(probabilities: np.ndarray, labels: list[str]) -> list[dict]:
        indices = np.argsort(probabilities)[::-1][:3]
        return [
            {
                "species": str(labels[index]),
                "species_id": normalize_species_id(str(labels[index])),
                "probability": round(float(probabilities[index]) * 100, 2),
            }
            for index in indices
        ]

    @staticmethod
    def _consensus(results: list[dict]) -> dict:
        if not results:
            return {
                "species": "Unavailable",
                "species_id": "unavailable",
                "agreement": "No models available",
                "votes": {},
            }
        votes: dict[str, int] = {}
        display_names: dict[str, str] = {}
        for result in results:
            species_id = result["species_id"]
            votes[species_id] = votes.get(species_id, 0) + 1
            display_names.setdefault(species_id, result["species"])
        winner, count = max(votes.items(), key=lambda item: item[1])
        return {
            "species": display_names[winner],
            "species_id": winner,
            "votes": votes,
            "agreement": "All models agree" if count == len(results) else f"Conflicting predictions ({count}/{len(results)} agree)",
        }

    def predict_audio_samples(self, audio: np.ndarray, sample_rate: int) -> dict:
        if not self.audio_models:
            raise RuntimeError("No audio classifiers are available. Train them with models/audio_model/train.py.")
        features = extract_audio_features(audio, sample_rate)
        results = []
        for name, (model, labels) in self.audio_models.items():
            started = time.perf_counter()
            probabilities = np.asarray(model.predict_proba(features)[0], dtype=float)
            elapsed = (time.perf_counter() - started) * 1000
            best = int(np.argmax(probabilities))
            label = str(labels[best])
            metrics = self.audio_metrics.get("models", {}).get(name, {})
            results.append({
                "model": name,
                "species": label,
                "species_id": normalize_species_id(label),
                "confidence": round(float(probabilities[best]) * 100, 2),
                "latency_ms": round(elapsed, 2),
                "top_three": self._top_three(probabilities, labels),
                "train_accuracy": metrics.get("train_accuracy"),
                "validation_accuracy": metrics.get("validation_accuracy"),
                "test_accuracy": metrics.get("test_accuracy"),
                "test_macro_f1": metrics.get("test_macro_f1"),
            })
        return {
            "modality": "audio",
            "sample_rate": AUDIO_SAMPLE_RATE,
            "results": results,
            "consensus": self._consensus(results),
        }

    def predict_audio(self, path: Path) -> dict:
        audio, sample_rate = librosa.load(path, mono=True)
        return self.predict_audio_samples(audio, sample_rate)

    def predict_image(self, path: Path) -> dict:
        image = Image.open(path).convert("RGB")
        tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        results, feature_extraction_ms = self._predict_image_tensor(tensor)
        return {
            "modality": "image",
            "device": str(DEVICE),
            "feature_extraction_ms": round(feature_extraction_ms, 2),
            "results": results,
            "consensus": self._consensus(results),
        }

    def _predict_image_tensor(
        self,
        tensor: torch.Tensor,
        include_probabilities: bool = False,
    ) -> tuple[list[dict], float]:
        if self.image_encoder is None or not self.image_models:
            raise RuntimeError("No image classifiers are available. Run models/image_model/train_classifiers.py.")
        started = time.perf_counter()
        with torch.inference_mode():
            features = self.image_encoder(tensor).cpu().numpy()
        feature_extraction_ms = (time.perf_counter() - started) * 1000
        results = []
        for name, (model, labels) in self.image_models.items():
            started = time.perf_counter()
            probabilities = np.asarray(model.predict_proba(features)[0], dtype=float)
            elapsed = (time.perf_counter() - started) * 1000
            best = int(np.argmax(probabilities))
            label = str(labels[best])
            metrics = self.image_metrics.get("models", {}).get(name, {})
            result = {
                "model": name,
                "species": label,
                "species_id": normalize_species_id(label),
                "confidence": round(float(probabilities[best]) * 100, 2),
                "latency_ms": round(elapsed, 2),
                "top_three": self._top_three(probabilities, labels),
                "train_accuracy": metrics.get("train_accuracy"),
                "validation_accuracy": metrics.get("validation_accuracy"),
                "test_accuracy": metrics.get("test_accuracy"),
                "test_macro_f1": metrics.get("test_macro_f1"),
            }
            if include_probabilities:
                result["probs"] = probabilities
                result["labels"] = labels
            results.append(result)
        return results, feature_extraction_ms

    def predict_image_from_pil(self, image: Image.Image) -> tuple[list[dict], float]:
        tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        return self._predict_image_tensor(tensor, include_probabilities=True)
