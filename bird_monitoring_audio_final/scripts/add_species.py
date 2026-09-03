from __future__ import annotations
import argparse, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import AUDIO_DATASET_DIR
from src.database.species_manager import SpeciesManager
from src.audio.generate_embeddings import generate_audio_embeddings

def normalize_species_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--common-name", default=None)
    parser.add_argument("--scientific-name", default="")
    parser.add_argument("--generate", action="store_true", help="Generate embeddings after adding files.")
    args = parser.parse_args()

    species_id = normalize_species_name(args.species)
    common_name = args.common_name or species_id.replace("_", " ").title()

    (AUDIO_DATASET_DIR / species_id).mkdir(parents=True, exist_ok=True)
    SpeciesManager().add(species_id, common_name, args.scientific_name)

    print(f"Created/registered species: {species_id}")
    print(f"Put recordings into: {AUDIO_DATASET_DIR / species_id}")

    if args.generate:
        generate_audio_embeddings()

if __name__ == "__main__":
    main()
