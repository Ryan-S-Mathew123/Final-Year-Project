"""Comparative bird-model evaluation dashboard.

Run from the repository root with:
    python models/flask_frontend/app.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
import shutil

import joblib
import librosa
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms

# Optional video-processing imports — gracefully degraded if not installed
try:
    import cv2 as _cv2  # noqa: F401 — just checking availability
    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False

try:
    try:
        from moviepy import VideoFileClip as _VideoFileClip  # moviepy >= 2.0
    except ImportError:
        from moviepy.editor import VideoFileClip as _VideoFileClip  # moviepy < 2.0
    _MOVIEPY_AVAILABLE = True
except ImportError:
    _MOVIEPY_AVAILABLE = False


FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parents[1]
AUDIO_DIR = REPO_ROOT / "models" / "audio_model"
IMAGE_DIR = REPO_ROOT / "models" / "image_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, template_folder=str(FRONTEND_DIR / "templates"), static_folder=str(FRONTEND_DIR / "static"))

AUDIO_MODELS = {
    "Random Forest": AUDIO_DIR / "model.pkl",
    "Gradient Boosting": AUDIO_DIR / "model_gradient_boosting.pkl",
    "SVM": AUDIO_DIR / "model_svm.pkl",
    "K-Nearest Neighbors": AUDIO_DIR / "model_knn.pkl",
}
IMAGE_CHECKPOINTS = {
    "ResNet50": IMAGE_DIR / "checkpoints" / "resnet50_best.pth",
    "EfficientNet-B0": IMAGE_DIR / "checkpoints" / "efficientnet_best.pth",
    "MobileNetV3-Large": IMAGE_DIR / "checkpoints" / "mobilenet_best.pth",
    "ConvNeXt-Tiny": IMAGE_DIR / "checkpoints" / "convnext_best.pth",
}
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ModelRegistry:
    """Loads every available trained artifact once and reuses it for requests."""

    def __init__(self) -> None:
        self.audio_models: dict[str, object] = {}
        self.audio_labels = None
        self.image_models: dict[str, tuple[torch.nn.Module, list[str]]] = {}
        self.load_errors: list[str] = []
        self._load_audio_models()
        self._load_image_models()

    def _load_audio_models(self) -> None:
        encoder_path = AUDIO_DIR / "label_encoder.pkl"
        try:
            self.audio_labels = joblib.load(encoder_path)
        except Exception as exc:
            self.load_errors.append(f"Audio label encoder: {exc}")
            return
        for name, path in AUDIO_MODELS.items():
            try:
                self.audio_models[name] = joblib.load(path)
            except Exception as exc:
                self.load_errors.append(f"{name}: {exc}")

    @staticmethod
    def _create_image_model(name: str, class_count: int) -> torch.nn.Module:
        if name == "ResNet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, class_count)
        elif name == "EfficientNet-B0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, class_count)
        elif name == "MobileNetV3-Large":
            model = models.mobilenet_v3_large(weights=None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, class_count)
        elif name == "ConvNeXt-Tiny":
            model = models.convnext_tiny(weights=None)
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, class_count)
        else:
            raise ValueError(f"Unsupported image model: {name}")
        return model

    def _load_image_models(self) -> None:
        for name, path in IMAGE_CHECKPOINTS.items():
            try:
                checkpoint = torch.load(path, map_location=DEVICE)
                classes = list(checkpoint["classes"])
                model = self._create_image_model(name, len(classes))
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(DEVICE).eval()
                self.image_models[name] = (model, classes)
            except Exception as exc:
                self.load_errors.append(f"{name}: {exc}")

    @staticmethod
    def _top_three(probabilities: np.ndarray, labels: list[str]) -> list[dict]:
        indices = np.argsort(probabilities)[::-1][:3]
        return [{"species": str(labels[i]), "probability": round(float(probabilities[i]) * 100, 2)} for i in indices]

    @staticmethod
    def _consensus(results: list[dict]) -> dict:
        if not results:
            return {"species": "Unavailable", "agreement": "No models available", "votes": {}}
        votes: dict[str, int] = {}
        for result in results:
            votes[result["species"]] = votes.get(result["species"], 0) + 1
        winner, count = max(votes.items(), key=lambda item: item[1])
        return {
            "species": winner,
            "votes": votes,
            "agreement": "All models agree" if count == len(results) else f"Conflicting predictions ({count}/{len(results)} agree)",
        }

    def predict_audio(self, path: Path) -> dict:
        audio, sample_rate = librosa.load(path, mono=True)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        features = np.mean(mfcc.T, axis=0).reshape(1, -1)
        results = []
        labels = list(self.audio_labels.classes_)
        for name, model in self.audio_models.items():
            started = time.perf_counter()
            probabilities = np.asarray(model.predict_proba(features)[0], dtype=float)
            elapsed = (time.perf_counter() - started) * 1000
            best = int(np.argmax(probabilities))
            results.append({
                "model": name,
                "species": str(labels[best]),
                "confidence": round(float(probabilities[best]) * 100, 2),
                "latency_ms": round(elapsed, 2),
                "top_three": self._top_three(probabilities, labels),
            })
        return {"modality": "audio", "sample_rate": sample_rate, "results": results, "consensus": self._consensus(results)}

    def predict_image(self, path: Path) -> dict:
        image = Image.open(path).convert("RGB")
        tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        results = []
        with torch.inference_mode():
            for name, (model, labels) in self.image_models.items():
                started = time.perf_counter()
                probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
                elapsed = (time.perf_counter() - started) * 1000
                best = int(np.argmax(probabilities))
                results.append({
                    "model": name,
                    "species": str(labels[best]),
                    "confidence": round(float(probabilities[best]) * 100, 2),
                    "latency_ms": round(elapsed, 2),
                    "top_three": self._top_three(probabilities, labels),
                })
        return {"modality": "image", "device": str(DEVICE), "results": results, "consensus": self._consensus(results)}

    def predict_image_from_pil(self, image: Image.Image) -> dict:
        """Run all image models on an already-opened PIL image."""
        tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        results = []
        with torch.inference_mode():
            for name, (model, labels) in self.image_models.items():
                started = time.perf_counter()
                probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
                elapsed = (time.perf_counter() - started) * 1000
                results.append({"model": name, "labels": labels, "probs": probabilities, "latency_ms": elapsed})
        return results

    def predict_video(self, path: Path, n_frames: int = 8) -> dict:
        """Extract audio + N frames from a video and run all models."""
        if not _OPENCV_AVAILABLE:
            raise RuntimeError("opencv-python is not installed. Run: pip install opencv-python")
        if not _MOVIEPY_AVAILABLE:
            raise RuntimeError("moviepy is not installed. Run: pip install moviepy")

        import cv2
        try:
            from moviepy import VideoFileClip  # moviepy >= 2.0
        except ImportError:
            from moviepy.editor import VideoFileClip  # moviepy < 2.0

        tmpdir = Path(mkdtemp())
        try:
            # ── video metadata ──────────────────────────────────────────────
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError("Could not open video file.")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            duration_s = total_frames / fps

            # ── frame extraction ────────────────────────────────────────────
            indices = np.linspace(0, max(total_frames - 1, 0), n_frames, dtype=int)
            per_model_probs: dict[str, list[np.ndarray]] = {}
            per_model_labels: dict[str, list[str]] = {}
            per_model_latencies: dict[str, list[float]] = {}
            frames_processed = 0
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_results = self.predict_image_from_pil(pil_image)
                for res in frame_results:
                    m = res["model"]
                    per_model_probs.setdefault(m, []).append(res["probs"])
                    per_model_labels[m] = res["labels"]
                    per_model_latencies.setdefault(m, []).append(res["latency_ms"])
                frames_processed += 1
            cap.release()

            # Average probabilities across frames
            image_results = []
            for name in per_model_probs:
                avg_probs = np.mean(per_model_probs[name], axis=0)
                labels = per_model_labels[name]
                best = int(np.argmax(avg_probs))
                image_results.append({
                    "model": name,
                    "species": str(labels[best]),
                    "confidence": round(float(avg_probs[best]) * 100, 2),
                    "latency_ms": round(float(np.mean(per_model_latencies[name])), 2),
                    "top_three": self._top_three(avg_probs, labels),
                })
            image_payload = {
                "modality": "image",
                "device": str(DEVICE),
                "results": image_results,
                "consensus": self._consensus(image_results),
            }

            # ── audio extraction ────────────────────────────────────────────
            audio_payload = None
            audio_path = tmpdir / "extracted_audio.wav"
            try:
                clip = VideoFileClip(str(path))
                if clip.audio is not None:
                    clip.audio.write_audiofile(str(audio_path), logger=None)
                    clip.close()
                    audio_payload = self.predict_audio(audio_path)
                else:
                    clip.close()
                    audio_payload = {"modality": "audio", "error": "No audio track found in video."}
            except Exception as audio_exc:
                audio_payload = {"modality": "audio", "error": str(audio_exc)}

            # ── multimodal fusion (majority vote) ───────────────────────────
            candidates = []
            if image_payload["consensus"]["species"] not in ("Unavailable",):
                candidates.append(image_payload["consensus"]["species"])
            if audio_payload and "consensus" in audio_payload and audio_payload["consensus"]["species"] not in ("Unavailable",):
                candidates.append(audio_payload["consensus"]["species"])
            fusion_votes: dict[str, int] = {}
            for s in candidates:
                fusion_votes[s] = fusion_votes.get(s, 0) + 1
            if fusion_votes:
                fusion_species = max(fusion_votes, key=lambda k: fusion_votes[k])
            else:
                fusion_species = "Unavailable"

            return {
                "audio": audio_payload,
                "image": image_payload,
                "multimodal": {
                    "species": fusion_species,
                    "votes": fusion_votes,
                    "agreement": "Both modalities agree" if len(set(candidates)) == 1 else "Modalities disagree — showing majority",
                },
                "frame_count": frames_processed,
                "duration_s": round(duration_s, 2),
                "fps": round(fps, 2),
            }
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


registry = ModelRegistry()


def _predict(modality: str):
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Attach a file using the 'file' field."}), 400
    suffix = Path(upload.filename).suffix.lower()
    allowed = {".wav", ".mp3", ".ogg", ".flac", ".m4a"} if modality == "audio" else {".jpg", ".jpeg", ".png", ".webp"}
    if suffix not in allowed:
        return jsonify({"error": f"Unsupported {modality} format: {suffix or 'missing extension'}"}), 400
    temporary = None
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            upload.save(handle)
            temporary = Path(handle.name)
        result = registry.predict_audio(temporary) if modality == "audio" else registry.predict_image(temporary)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"{modality.title()} inference failed: {exc}"}), 500
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


@app.errorhandler(Exception)
def handle_any_exception(exc):
    """Always return JSON instead of Flask's default HTML error pages."""
    import traceback
    print("[/predict/video] Unhandled exception:", traceback.format_exc())
    return jsonify({"error": str(exc)}), 500


@app.get("/")
def index():
    return render_template("index.html", device=str(DEVICE), load_errors=registry.load_errors)


@app.get("/video")
def video_page():
    return render_template(
        "video.html",
        device=str(DEVICE),
        load_errors=registry.load_errors,
        opencv_available=_OPENCV_AVAILABLE,
        moviepy_available=_MOVIEPY_AVAILABLE,
    )


@app.post("/predict/audio")
def predict_audio():
    return _predict("audio")


@app.post("/predict/image")
def predict_image():
    return _predict("image")


@app.post("/predict/video")
def predict_video():
    if not _OPENCV_AVAILABLE or not _MOVIEPY_AVAILABLE:
        missing = []
        if not _OPENCV_AVAILABLE:
            missing.append("opencv-python")
        if not _MOVIEPY_AVAILABLE:
            missing.append("moviepy")
        return jsonify({"error": f"Missing dependencies: {', '.join(missing)}. Run: pip install {' '.join(missing)}"}), 500
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Attach a video file using the 'file' field."}), 400
    suffix = Path(upload.filename).suffix.lower()
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if suffix not in allowed:
        return jsonify({"error": f"Unsupported video format: {suffix or 'missing extension'}"}), 400
    n_frames = min(int(request.form.get("n_frames", 8)), 32)
    temporary = None
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            upload.save(handle)
            temporary = Path(handle.name)
        result = registry.predict_video(temporary, n_frames=n_frames)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Video inference failed: {exc}"}), 500
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "audio_models_loaded": list(registry.audio_models),
        "image_models_loaded": list(registry.image_models),
        "load_errors": registry.load_errors,
        "device": str(DEVICE),
        "video_support": {
            "opencv": _OPENCV_AVAILABLE,
            "moviepy": _MOVIEPY_AVAILABLE,
        },
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
