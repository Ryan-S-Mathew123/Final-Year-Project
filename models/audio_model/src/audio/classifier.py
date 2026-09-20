from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np
from config import AUDIO_SEARCH_MODE, AUDIO_UNKNOWN_THRESHOLD, CLUSTER_MIN_SAMPLES, TOP_K
from src.audio.encoder import AudioEncoder
from src.audio.low_confidence_collector import LowConfidenceCollector
from src.database.metadata_manager import MetadataManager
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class AudioClassifier:
    def __init__(self, mode: Literal["individual", "prototype"] = AUDIO_SEARCH_MODE) -> None:
        if mode not in ("individual", "prototype"):
            raise ValueError("mode must be 'individual' or 'prototype'")
        self.mode = mode
        self.encoder = AudioEncoder()
        self.metadata_manager = MetadataManager()
        # Share the same encoder instance with the collector to avoid loading CLAP twice
        self._collector: LowConfidenceCollector | None = None

    @property
    def collector(self) -> LowConfidenceCollector:
        if self._collector is None:
            self._collector = LowConfidenceCollector(encoder=self.encoder)
        return self._collector

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

        result = {
            "species": "UNKNOWN" if is_unknown else best_result["species"],
            "confidence": float(max(0.0, best_similarity)),
            "similarity": best_similarity,
            "is_unknown": is_unknown,
            "top_k": top_results,
            "low_confidence_saved": False,
        }

        # ── Auto-save low-confidence predictions ──
        if is_unknown:
            try:
                saved_record = self.collector.save_recording(path, result)
                result["low_confidence_saved"] = True
                result["low_confidence_sample_id"] = saved_record["sample_id"]
                logger.info(
                    "Low-confidence prediction auto-saved: %s (confidence=%.3f)",
                    saved_record["sample_id"][:12],
                    result["confidence"],
                )

                # Auto-cluster if we have enough samples
                pending_count = self.collector.count()
                if pending_count >= CLUSTER_MIN_SAMPLES:
                    logger.info(
                        "Reached %d low-confidence samples — auto-clustering...",
                        pending_count,
                    )
                    cluster_result = self.collector.cluster()
                    result["auto_clustered"] = True
                    result["cluster_info"] = cluster_result
            except Exception as e:
                logger.error("Failed to save low-confidence recording: %s", e)

        return result

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
