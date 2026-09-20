"""
Low-confidence auto-record, embed & cluster pipeline.

When the classifier returns confidence < AUDIO_UNKNOWN_THRESHOLD (0.7),
audio clips are automatically saved, embedded via CLAP, and periodically
clustered using agglomerative clustering so that similar unknowns are
grouped together for manual labelling.
"""
from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from config import (
    AUDIO_DATASET_DIR,
    AUDIO_EMBEDDINGS_DIR,
    CLUSTER_MAX_DISTANCE,
    CLUSTER_MIN_SAMPLES,
    LOW_CONFIDENCE_DIR,
    LOW_CONFIDENCE_EMBEDDINGS_DIR,
    LOW_CONFIDENCE_METADATA_PATH,
)
from src.audio.encoder import AudioEncoder
from src.database.metadata_manager import MetadataManager
from src.database.species_manager import SpeciesManager
from src.utils.helpers import get_logger, load_json, save_json, sha256_file

logger = get_logger(__name__)


class LowConfidenceCollector:
    """Collects, embeds, clusters, and labels low-confidence predictions."""

    def __init__(self, encoder: AudioEncoder | None = None) -> None:
        # Lazy-load the encoder only when we actually need to embed
        self._encoder = encoder
        LOW_CONFIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        LOW_CONFIDENCE_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        LOW_CONFIDENCE_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    @property
    def encoder(self) -> AudioEncoder:
        if self._encoder is None:
            self._encoder = AudioEncoder()
        return self._encoder

    # ── persistence helpers ──────────────────────────────────────────

    def _load_samples(self) -> list[dict[str, Any]]:
        return load_json(LOW_CONFIDENCE_METADATA_PATH, [])

    def _save_samples(self, samples: list[dict[str, Any]]) -> None:
        save_json(LOW_CONFIDENCE_METADATA_PATH, samples)

    # ── save recording ───────────────────────────────────────────────

    def save_recording(
        self,
        audio_path: Path,
        prediction_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Copy audio to low-confidence store, compute CLAP embedding, write metadata."""
        audio_path = Path(audio_path)
        file_hash = sha256_file(audio_path)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sample_id = hashlib.sha256(
            f"low_conf:{file_hash}:{timestamp}".encode()
        ).hexdigest()

        # Copy audio file
        dest_filename = f"{timestamp}_{sample_id[:12]}{audio_path.suffix}"
        dest_path = LOW_CONFIDENCE_DIR / dest_filename
        shutil.copy2(audio_path, dest_path)

        # Generate CLAP embedding
        embedding = self.encoder.encode(dest_path)
        embedding_path = LOW_CONFIDENCE_EMBEDDINGS_DIR / f"{sample_id}.npy"
        np.save(embedding_path, embedding)

        # Build metadata record
        record = {
            "sample_id": sample_id,
            "source_path": str(audio_path.resolve()),
            "saved_path": str(dest_path.resolve()),
            "embedding_path": str(embedding_path.resolve()),
            "file_hash": file_hash,
            "original_prediction": prediction_result.get("species", "UNKNOWN"),
            "confidence": float(prediction_result.get("confidence", 0.0)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cluster_id": None,
            "label": None,
        }

        samples = self._load_samples()
        samples.append(record)
        self._save_samples(samples)

        logger.info(
            "Saved low-confidence recording: %s (confidence=%.3f)",
            dest_filename,
            record["confidence"],
        )
        return record

    # ── count ────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the number of unlabelled low-confidence recordings."""
        return sum(1 for s in self._load_samples() if s.get("label") is None)

    # ── clustering ───────────────────────────────────────────────────

    def cluster(
        self,
        distance_threshold: float = CLUSTER_MAX_DISTANCE,
    ) -> dict[str, Any]:
        """
        Load all unlabelled low-confidence embeddings, run agglomerative
        clustering (cosine affinity), and assign cluster IDs to metadata.
        """
        samples = self._load_samples()
        unlabelled = [s for s in samples if s.get("label") is None]

        if len(unlabelled) < 2:
            logger.warning("Not enough unlabelled samples to cluster (%d)", len(unlabelled))
            return {"n_clusters": 0, "cluster_sizes": {}}

        # Load embeddings
        embeddings = []
        valid_indices = []
        for idx, sample in enumerate(unlabelled):
            emb_path = Path(sample["embedding_path"])
            if emb_path.exists():
                embeddings.append(np.load(emb_path).astype(np.float32))
                valid_indices.append(idx)
            else:
                logger.warning("Embedding missing for %s", sample["sample_id"])

        if len(embeddings) < 2:
            return {"n_clusters": 0, "cluster_sizes": {}}

        embedding_matrix = np.vstack(embeddings)

        # Cosine distance matrix = 1 - cosine_similarity
        # Since embeddings are L2-normalised, cosine_sim = dot product
        similarity_matrix = embedding_matrix @ embedding_matrix.T
        distance_matrix = 1.0 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0.0)  # eliminate floating-point noise
        distance_matrix = np.clip(distance_matrix, 0.0, 2.0)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        labels = clustering.fit_predict(distance_matrix)

        # Write cluster IDs back to metadata
        sample_id_to_cluster = {}
        for i, vi in enumerate(valid_indices):
            cluster_id = int(labels[i])
            unlabelled[vi]["cluster_id"] = cluster_id
            sample_id_to_cluster[unlabelled[vi]["sample_id"]] = cluster_id

        for sample in samples:
            if sample["sample_id"] in sample_id_to_cluster:
                sample["cluster_id"] = sample_id_to_cluster[sample["sample_id"]]
        self._save_samples(samples)

        cluster_sizes = dict(Counter(int(l) for l in labels))
        n_clusters = len(cluster_sizes)
        logger.info("Clustered %d samples into %d clusters", len(embeddings), n_clusters)

        return {"n_clusters": n_clusters, "cluster_sizes": cluster_sizes}

    # ── review clusters ──────────────────────────────────────────────

    def get_clusters_summary(self) -> dict[int, dict[str, Any]]:
        """
        Returns a dict mapping cluster_id → {count, avg_similarity, samples}.
        """
        samples = self._load_samples()
        unlabelled = [s for s in samples if s.get("label") is None and s.get("cluster_id") is not None]

        clusters: dict[int, list[dict]] = {}
        for s in unlabelled:
            clusters.setdefault(s["cluster_id"], []).append(s)

        summary = {}
        for cluster_id, members in sorted(clusters.items()):
            # Compute average intra-cluster cosine similarity
            embeddings = []
            for m in members:
                emb_path = Path(m["embedding_path"])
                if emb_path.exists():
                    embeddings.append(np.load(emb_path).astype(np.float32))

            avg_sim = 0.0
            if len(embeddings) >= 2:
                mat = np.vstack(embeddings)
                sims = mat @ mat.T
                # Average of upper triangle (excluding diagonal)
                n = len(embeddings)
                avg_sim = float((sims.sum() - n) / (n * (n - 1))) if n > 1 else 0.0

            summary[cluster_id] = {
                "count": len(members),
                "avg_similarity": round(avg_sim, 4),
                "samples": [
                    {
                        "sample_id": m["sample_id"],
                        "saved_path": m["saved_path"],
                        "confidence": m["confidence"],
                        "original_prediction": m["original_prediction"],
                    }
                    for m in members
                ],
            }

        return summary

    # ── label a cluster ──────────────────────────────────────────────

    def label_cluster(self, cluster_id: int, species_id: str) -> int:
        """
        Promote all samples in a cluster to the training dataset.

        - Moves audio files into datasets/audio/<species_id>/
        - Generates standard embeddings via the main pipeline
        - Updates main metadata
        - Removes from low-confidence metadata

        Returns the number of samples promoted.
        """
        species_dir = AUDIO_DATASET_DIR / species_id
        species_dir.mkdir(parents=True, exist_ok=True)

        # Register the species if new
        SpeciesManager().add(species_id, species_id.replace("_", " ").title())

        samples = self._load_samples()
        to_promote = [s for s in samples if s.get("cluster_id") == cluster_id and s.get("label") is None]

        if not to_promote:
            logger.warning("No unlabelled samples found in cluster %d", cluster_id)
            return 0

        metadata_manager = MetadataManager()
        promoted_count = 0

        for sample in to_promote:
            src = Path(sample["saved_path"])
            if not src.exists():
                logger.warning("Audio file missing: %s", src)
                continue

            # Move audio into training dataset
            dest = species_dir / src.name
            shutil.move(str(src), str(dest))

            # Generate a new standard embedding
            file_hash = sha256_file(dest)
            std_sample_id = hashlib.sha256(
                f"audio:{species_id}:{file_hash}".encode()
            ).hexdigest()
            std_embedding_path = AUDIO_EMBEDDINGS_DIR / f"{std_sample_id}.npy"

            if not std_embedding_path.exists():
                embedding = self.encoder.encode(dest)
                np.save(std_embedding_path, embedding)

            metadata_manager.upsert_sample({
                "sample_id": std_sample_id,
                "species_id": species_id,
                "source_path": str(dest.resolve()),
                "file_hash": file_hash,
                "embedding_path": str(std_embedding_path.resolve()),
            })

            # Clean up the low-confidence embedding
            lc_emb = Path(sample["embedding_path"])
            if lc_emb.exists():
                lc_emb.unlink()

            sample["label"] = species_id
            promoted_count += 1

        self._save_samples(samples)

        # Rebuild prototypes with the new data
        metadata_manager.rebuild_prototypes()
        SpeciesManager().update_audio_count(
            species_id,
            sum(1 for s in metadata_manager.samples_with_embeddings() if s["species_id"] == species_id),
        )

        logger.info("Promoted %d samples from cluster %d → %s", promoted_count, cluster_id, species_id)
        return promoted_count

    # ── discard a cluster ────────────────────────────────────────────

    def discard_cluster(self, cluster_id: int) -> int:
        """Delete audio + embeddings for a cluster (noise / not birds). Returns count discarded."""
        samples = self._load_samples()
        to_discard = [s for s in samples if s.get("cluster_id") == cluster_id and s.get("label") is None]

        if not to_discard:
            logger.warning("No unlabelled samples found in cluster %d", cluster_id)
            return 0

        discarded = 0
        for sample in to_discard:
            for key in ("saved_path", "embedding_path"):
                p = Path(sample[key])
                if p.exists():
                    p.unlink()
            sample["label"] = "__discarded__"
            discarded += 1

        self._save_samples(samples)
        logger.info("Discarded %d samples from cluster %d", discarded, cluster_id)
        return discarded
