import os
import librosa
import numpy as np
import joblib
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import AUDIO_UNKNOWN_THRESHOLD, MODEL_MARGIN_THRESHOLD

app = FastAPI(title="Bird Audio Classifier")

# Setup templates
templates = Jinja2Templates(directory="templates")

# ── Load the trained RandomForest model ──
MODEL_PATH = "model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

model = None
le = None

if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    print(f"Model loaded: {len(le.classes_)} classes")
else:
    print("Warning: Model or Label Encoder not found. Please run train.py first.")

# ── Low-confidence threshold ──
CONFIDENCE_THRESHOLD = AUDIO_UNKNOWN_THRESHOLD


def should_mark_low_confidence(probabilities, threshold: float = CONFIDENCE_THRESHOLD, min_margin: float = MODEL_MARGIN_THRESHOLD) -> bool:
    """Return True when the prediction is either too weak or not decisively separated."""
    if not isinstance(probabilities, (list, tuple, np.ndarray)) or len(probabilities) == 0:
        return True

    probabilities = np.asarray(probabilities, dtype=np.float64)
    top_confidence = float(np.max(probabilities))
    second_confidence = float(np.partition(probabilities, -2)[-2]) if len(probabilities) > 1 else 0.0
    margin = top_confidence - second_confidence

    return top_confidence < threshold or margin < min_margin


# Lazy-loaded collector (loads CLAP only when first low-confidence prediction happens)
_collector = None
_collector_lock = threading.Lock()


def get_collector():
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                from src.audio.low_confidence_collector import LowConfidenceCollector
                from config import ensure_directories, CLUSTER_MIN_SAMPLES
                ensure_directories()
                _collector = LowConfidenceCollector()
    return _collector


def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}. Details: {e}")
        return None


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    if not model or not le:
        return {"error": "Model not trained yet. Run train.py first."}

    # Save uploaded file temporarily
    temp_file = f"temp_{audio.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    try:
        # Extract MFCC features
        features = extract_features(temp_file)
        if features is None:
            return {"error": "Failed to process audio file."}

        features = features.reshape(1, -1)

        # Predict with RandomForest
        prediction_encoded = model.predict(features)
        prediction_label = le.inverse_transform(prediction_encoded)[0]

        # Get confidence from probabilities
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))  # 0.0 to 1.0
        is_low_confidence = should_mark_low_confidence(probabilities, threshold=CONFIDENCE_THRESHOLD)

        result = {
            "prediction": prediction_label,
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_raw": confidence,
            "is_low_confidence": is_low_confidence,
            "low_confidence_saved": False,
            "threshold": CONFIDENCE_THRESHOLD,
            "margin": float(np.max(probabilities) - np.partition(probabilities, -2)[-2]) if len(probabilities) > 1 else 0.0,
        }

        # ── Auto-save if the classification is weak or ambiguous ──
        if is_low_confidence:
            try:
                collector = get_collector()
                prediction_info = {
                    "species": prediction_label,
                    "confidence": confidence,
                }
                saved_record = collector.save_recording(Path(temp_file), prediction_info)
                result["low_confidence_saved"] = True
                result["low_confidence_sample_id"] = saved_record["sample_id"][:12]

                # Auto-cluster if enough samples
                from config import CLUSTER_MIN_SAMPLES
                pending = collector.count()
                if pending >= CLUSTER_MIN_SAMPLES:
                    cluster_result = collector.cluster()
                    result["auto_clustered"] = True
                    result["cluster_info"] = cluster_result
                    print(f"Auto-clustered {pending} low-confidence samples into {cluster_result['n_clusters']} clusters")
            except Exception as e:
                print(f"Low-confidence save failed (non-fatal): {e}")

        return result
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)


# ── Low-confidence pipeline endpoints ────────────────────────────────


@app.get("/low-confidence/stats")
async def low_confidence_stats():
    """Return the count of pending low-confidence recordings and cluster summary."""
    collector = get_collector()
    return {
        "pending_count": collector.count(),
        "clusters": collector.get_clusters_summary(),
    }


@app.post("/low-confidence/cluster")
async def trigger_clustering():
    """Manually trigger re-clustering of all unlabelled low-confidence recordings."""
    collector = get_collector()
    pending = collector.count()
    if pending < 2:
        raise HTTPException(status_code=400, detail=f"Need at least 2 samples to cluster (have {pending})")
    result = collector.cluster()
    return {"status": "clustered", **result}


class LabelRequest(BaseModel):
    cluster_id: int
    species_id: str


@app.post("/low-confidence/label")
async def label_cluster(data: LabelRequest):
    """Label a cluster with a species ID, promoting samples to the training dataset."""
    collector = get_collector()
    count = collector.label_cluster(data.cluster_id, data.species_id)
    if count == 0:
        raise HTTPException(status_code=404, detail=f"No unlabelled samples in cluster {data.cluster_id}")
    return {
        "status": "labelled",
        "promoted_count": count,
        "species_id": data.species_id,
        "message": f"Promoted {count} samples to '{data.species_id}'. Run train.py to retrain.",
    }


class DiscardRequest(BaseModel):
    cluster_id: int


@app.post("/low-confidence/discard")
async def discard_cluster(data: DiscardRequest):
    """Discard a cluster (noise / non-bird audio)."""
    collector = get_collector()
    count = collector.discard_cluster(data.cluster_id)
    if count == 0:
        raise HTTPException(status_code=404, detail=f"No unlabelled samples in cluster {data.cluster_id}")
    return {
        "status": "discarded",
        "discarded_count": count,
    }
