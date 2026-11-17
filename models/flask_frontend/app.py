"""Comparative bird-model evaluation dashboard.

Run from the repository root with:
    python models/flask_frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
import shutil

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

from models.prediction_registry import ModelRegistry as SharedModelRegistry
from models.species import normalize_species_id


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, template_folder=str(FRONTEND_DIR / "templates"), static_folder=str(FRONTEND_DIR / "static"))

class ModelRegistry(SharedModelRegistry):
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
                    "species_id": normalize_species_id(str(labels[best])),
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
            species_names: dict[str, str] = {}
            image_consensus = image_payload["consensus"]
            if image_consensus["species_id"] != "unavailable":
                candidates.append(image_consensus["species_id"])
                species_names[image_consensus["species_id"]] = image_consensus["species"]
            if audio_payload and "consensus" in audio_payload:
                audio_consensus = audio_payload["consensus"]
                if audio_consensus["species_id"] != "unavailable":
                    candidates.append(audio_consensus["species_id"])
                    species_names.setdefault(audio_consensus["species_id"], audio_consensus["species"])
            fusion_votes: dict[str, int] = {}
            for species_id in candidates:
                fusion_votes[species_id] = fusion_votes.get(species_id, 0) + 1
            if fusion_votes:
                fusion_species_id = max(fusion_votes, key=lambda species_id: fusion_votes[species_id])
                fusion_species = species_names[fusion_species_id]
            else:
                fusion_species_id = "unavailable"
                fusion_species = "Unavailable"

            return {
                "audio": audio_payload,
                "image": image_payload,
                "multimodal": {
                    "species": fusion_species,
                    "species_id": fusion_species_id,
                    "votes": fusion_votes,
                    "agreement": "Both modalities agree" if len(candidates) == 2 and len(set(candidates)) == 1 else "Modalities disagree — showing majority",
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
