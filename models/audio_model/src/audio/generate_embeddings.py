from __future__ import annotations
import hashlib
from collections import Counter
from config import AUDIO_DATASET_DIR, AUDIO_EMBEDDINGS_DIR, AUDIO_EXTENSIONS
from src.audio.encoder import AudioEncoder
from src.audio.preprocess import validate_audio
from src.database.metadata_manager import MetadataManager
from src.database.species_manager import SpeciesManager
from src.utils.helpers import get_logger, sha256_file
import numpy as np

logger = get_logger(__name__)

def generate_audio_embeddings() -> int:
    AUDIO_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    encoder = AudioEncoder()
    metadata_manager = MetadataManager()
    species_manager = SpeciesManager()
    processed = 0
    species_counts = Counter()

    if not AUDIO_DATASET_DIR.exists():
        raise RuntimeError(f"Audio dataset directory does not exist: {AUDIO_DATASET_DIR}")

    species_directories = sorted(path for path in AUDIO_DATASET_DIR.iterdir() if path.is_dir())

    for species_directory in species_directories:
        species_id = species_directory.name
        species_manager.add(species_id, species_id.replace("_", " ").title())
        logger.info("Processing species: %s", species_id)

        audio_files = sorted(
            path for path in species_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
        logger.info("%s: found %d audio files", species_id, len(audio_files))

        for audio_path in audio_files:
            valid, reason, _ = validate_audio(audio_path)
            if not valid:
                logger.warning("Skipping %s: %s", audio_path, reason)
                continue

            try:
                file_hash = sha256_file(audio_path)
                sample_id = hashlib.sha256(f"audio:{species_id}:{file_hash}".encode("utf-8")).hexdigest()
                embedding_path = AUDIO_EMBEDDINGS_DIR / f"{sample_id}.npy"

                if not embedding_path.exists():
                    embedding = encoder.encode(audio_path)
                    np.save(embedding_path, embedding)
                    logger.info("Generated embedding: %s", audio_path.name)
                else:
                    logger.info("Embedding already exists: %s", audio_path.name)

                metadata_manager.upsert_sample({
                    "sample_id": sample_id,
                    "species_id": species_id,
                    "source_path": str(audio_path.resolve()),
                    "file_hash": file_hash,
                    "embedding_path": str(embedding_path.resolve()),
                })
                processed += 1
                species_counts[species_id] += 1
            except Exception as error:
                logger.exception("Failed to process %s: %s", audio_path, error)

    metadata_manager.rebuild_prototypes()
    for species_id, count in species_counts.items():
        species_manager.update_audio_count(species_id, count)

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION SUMMARY")
    print("=" * 60)
    for species_id in sorted(species_counts):
        print(f"{species_id}: {species_counts[species_id]} embeddings")
    print("-" * 60)
    print(f"TOTAL: {processed} audio embeddings")
    print("=" * 60)
    return processed
