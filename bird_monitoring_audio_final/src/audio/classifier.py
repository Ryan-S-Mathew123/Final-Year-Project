from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np
from config import AUDIO_SEARCH_MODE, AUDIO_UNKNOWN_THRESHOLD, TOP_K
from src.audio.encoder import AudioEncoder
from src.database.metadata_manager import MetadataManager

class AudioClassifier:
    def __init__(self, mode: Literal["individual", "prototype"] = AUDIO_SEARCH_MODE) -> None:
        if mode not in ("individual", "prototype"):
            raise ValueError("mode must be 'individual' or 'prototype'")
        self.mode = mode
        self.encoder = AudioEncoder()
        self.metadata_manager = MetadataManager()

    def predict(self, path: Path, top_k: int = TOP_K, threshold: float = AUDIO_UNKNOWN_THRESHOLD) -> dict:
        query_embedding = self.encoder.encode(path)
        records = self._get_records()
        if not records:
            raise RuntimeError("No embeddings found. Run generate_audio_embeddings.py first.")

        embeddings = np.vstack([record["embedding"] for record in records])
        similarities = embeddings @ query_embedding
        sorted_indices = np.argsort(similarities)[::-1][:top_k]

        top_results = [
            {"species": records[index]["species_id"], "similarity": float(similarities[index])}
            for index in sorted_indices
        ]

        best_result = top_results[0]
        best_similarity = best_result["similarity"]
        is_unknown = best_similarity < threshold

        return {
            "species": "UNKNOWN" if is_unknown else best_result["species"],
            "confidence": float(max(0.0, best_similarity)),
            "similarity": best_similarity,
            "is_unknown": is_unknown,
            "top_k": top_results,
        }

    def _get_records(self) -> list[dict]:
        if self.mode == "prototype":
            prototypes = self.metadata_manager.load_prototypes()
            return [
                {"species_id": species_id, "embedding": np.asarray(embedding, dtype=np.float32)}
                for species_id, embedding in prototypes.items()
            ]

        records = []
        for sample in self.metadata_manager.samples_with_embeddings():
            records.append({
                "species_id": sample["species_id"],
                "embedding": np.load(sample["embedding_path"]).astype(np.float32),
            })
        return records
