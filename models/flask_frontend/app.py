"""Comparative bird-model evaluation dashboard.

Run from the repository root with:
    python models/flask_frontend/app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
import shutil

import joblib
import librosa
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.image_model.feature_extractor import IMAGE_TRANSFORM, load_image_encoder
from models.model_families import MODEL_ARTIFACTS


AUDIO_DIR = REPO_ROOT / "models" / "audio_model"
IMAGE_DIR = REPO_ROOT / "models" / "image_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, template_folder=str(FRONTEND_DIR / "templates"), static_folder=str(FRONTEND_DIR / "static"))

class ModelRegistry:
    """Loads the same classifier families for both modalities."""

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
        if not self.audio_models:
            raise RuntimeError("No audio classifiers are available. Train them with models/audio_model/train.py.")
        audio, sample_rate = librosa.load(path, mono=True)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        features = np.mean(mfcc.T, axis=0).reshape(1, -1)
        results = []
        for name, (model, labels) in self.audio_models.items():
            started = time.perf_counter()
            probabilities = np.asarray(model.predict_proba(features)[0], dtype=float)
            elapsed = (time.perf_counter() - started) * 1000
            best = int(np.argmax(probabilities))
            metrics = self.audio_metrics.get("models", {}).get(name, {})
            results.append({
                "model": name,
                "species": str(labels[best]),
                "confidence": round(float(probabilities[best]) * 100, 2),
                "latency_ms": round(elapsed, 2),
                "top_three": self._top_three(probabilities, labels),
                "train_accuracy": metrics.get("train_accuracy"),
                "validation_accuracy": metrics.get("validation_accuracy"),
                "test_accuracy": metrics.get("test_accuracy"),
                "test_macro_f1": metrics.get("test_macro_f1"),
            })
        return {"modality": "audio", "sample_rate": sample_rate, "results": results, "consensus": self._consensus(results)}

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
            metrics = self.image_metrics.get("models", {}).get(name, {})
            result = {
                "model": name,
                "species": str(labels[best]),
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
        """Run all image classifiers on an already-opened PIL image."""
        tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        return self._predict_image_tensor(tensor, include_probabilities=True)

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
            per_model_metrics: dict[str, dict] = {}
            feature_extraction_times: list[float] = []
            frames_processed = 0
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_results, feature_extraction_ms = self.predict_image_from_pil(pil_image)
                feature_extraction_times.append(feature_extraction_ms)
                for res in frame_results:
                    m = res["model"]
                    per_model_probs.setdefault(m, []).append(res["probs"])
                    per_model_labels[m] = res["labels"]
                    per_model_latencies.setdefault(m, []).append(res["latency_ms"])
                    per_model_metrics[m] = {
                        "train_accuracy": res["train_accuracy"],
                        "validation_accuracy": res["validation_accuracy"],
                        "test_accuracy": res["test_accuracy"],
                        "test_macro_f1": res["test_macro_f1"],
                    }
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
                    **per_model_metrics[name],
                })
            image_payload = {
                "modality": "image",
                "device": str(DEVICE),
                "feature_extraction_ms": round(sum(feature_extraction_times), 2),
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
