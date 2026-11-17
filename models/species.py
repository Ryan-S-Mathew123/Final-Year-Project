"""Canonical species identifiers shared by modality-specific classifiers."""
from __future__ import annotations

import re


_ALIASES = {
    "grey_indian_hornbill": "indian_grey_hornbill",
    "grey_junglefowl_clips": "grey_junglefowl",
    "indian_peafoul": "indian_peafowl",
    "peregrine_falcon_clips": "peregrine_falcon",
    "rock_eagle": "rock_eagle_owl",
    "spotted_dove_": "spotted_dove",
}


def normalize_species_id(label: str) -> str:
    """Convert model-specific labels to the shared, stable species ID."""
    normalized = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return _ALIASES.get(normalized, normalized)
