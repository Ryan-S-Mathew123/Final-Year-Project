from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.prediction_registry import ModelRegistry


class ModelClassifier:
    """API-facing adapter over the shared comparative model registry."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    @staticmethod
    def _summarize(payload: dict) -> dict:
        consensus = payload["consensus"]
        species_id = consensus["species_id"]
        winning_models = [
            result for result in payload["results"]
            if result["species_id"] == species_id
        ]
        confidence = (
            float(np.mean([result["confidence"] for result in winning_models])) / 100
            if winning_models
            else 0.0
        )
        return {
            "species": consensus["species"],
            "species_id": species_id,
            "confidence": confidence,
            "model": "shared_comparison_families",
            "models": {
                result["model"]: {
                    "species": result["species"],
                    "species_id": result["species_id"],
                    "confidence": result["confidence"] / 100,
                }
                for result in payload["results"]
            },
            "agreement": consensus["agreement"],
            "votes": consensus["votes"],
        }

    def classify_audio(self, path: Path) -> dict:
        return self._summarize(self.registry.predict_audio(path))

    def classify_image(self, path: Path) -> dict:
        payload = self.registry.predict_image(path)
        return {
            **self._summarize(payload),
            "device": payload["device"],
        }


class BirdClassifier:
    def __init__(self):
        print("Bird classifier placeholder initialized")

    def predict(self, audio, sample_rate=44100):
        if audio is None or len(audio)==0:
            return {"species":"No audio","confidence":0.0,"detected":False}
        rms=float(np.sqrt(np.mean(np.square(audio))))
        if rms>0.01:
            return {"species":"Bird detected (placeholder)","confidence":0.85,"detected":True}
        return {"species":"No bird detected","confidence":0.0,"detected":False}
