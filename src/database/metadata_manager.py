from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from config import METADATA_DIR
from src.utils.helpers import load_json, save_json

SAMPLES_PATH = METADATA_DIR / "audio_samples.json"
PROTOTYPES_PATH = METADATA_DIR / "audio_prototypes.json"

class MetadataManager:
    def load_samples(self) -> list[dict[str, Any]]:
        return load_json(SAMPLES_PATH, [])

    def save_samples(self, samples: list[dict[str, Any]]) -> None:
        save_json(SAMPLES_PATH, samples)

    def upsert_sample(self, record: dict[str, Any]) -> None:
        samples = [sample for sample in self.load_samples() if sample["sample_id"] != record["sample_id"]]
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        samples.append(record)
        self.save_samples(samples)

    def samples_with_embeddings(self) -> list[dict[str, Any]]:
        return [sample for sample in self.load_samples() if Path(sample["embedding_path"]).exists()]

    def rebuild_prototypes(self) -> dict[str, list[float]]:
        grouped: dict[str, list[np.ndarray]] = {}
        for record in self.samples_with_embeddings():
            embedding = np.load(record["embedding_path"]).astype(np.float32)
            grouped.setdefault(record["species_id"], []).append(embedding)
        prototypes = {}
        for species_id, vectors in grouped.items():
            prototype = np.mean(np.vstack(vectors), axis=0)
            norm = np.linalg.norm(prototype)
            if norm > 0:
                prototype = prototype / norm
            prototypes[species_id] = prototype.astype(float).tolist()
        save_json(PROTOTYPES_PATH, prototypes)
        return prototypes

    def load_prototypes(self) -> dict[str, list[float]]:
        return load_json(PROTOTYPES_PATH, {})
