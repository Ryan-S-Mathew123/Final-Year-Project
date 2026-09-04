from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from config import METADATA_DIR, SPECIES
from src.utils.helpers import load_json, save_json

SPECIES_DB_PATH = METADATA_DIR / "species.json"

class SpeciesManager:
    def __init__(self) -> None:
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        if not SPECIES_DB_PATH.exists():
            database = {
                species_id: self._create_record(species_id, info["common_name"], info["scientific_name"])
                for species_id, info in SPECIES.items()
            }
            save_json(SPECIES_DB_PATH, database)

    def _create_record(self, species_id: str, common_name: str, scientific_name: str) -> dict[str, Any]:
        return {
            "species_id": species_id,
            "common_name": common_name,
            "scientific_name": scientific_name,
            "audio_embedding_count": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def all(self) -> dict[str, dict[str, Any]]:
        return load_json(SPECIES_DB_PATH, {})

    def exists(self, species_id: str) -> bool:
        return species_id in self.all()

    def add(self, species_id: str, common_name: str, scientific_name: str = "") -> None:
        database = self.all()
        if species_id in database:
            return
        database[species_id] = self._create_record(species_id, common_name, scientific_name)
        save_json(SPECIES_DB_PATH, database)

    def update_audio_count(self, species_id: str, count: int) -> None:
        database = self.all()
        if species_id not in database:
            self.add(species_id, species_id.replace("_", " ").title())
            database = self.all()
        database[species_id]["audio_embedding_count"] = count
        save_json(SPECIES_DB_PATH, database)
