from __future__ import annotations
import json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import AUDIO_DATASET_DIR, AUDIO_EXTENSIONS, MIN_AUDIO_DURATION_SECONDS, MAX_AUDIO_DURATION_SECONDS
from src.audio.preprocess import validate_audio
from src.utils.helpers import sha256_file

def main() -> None:
    report = {"species": {}, "summary": {"total": 0, "valid": 0, "invalid": 0, "duplicates": 0}}
    seen_hashes = {}

    for species_directory in sorted(AUDIO_DATASET_DIR.iterdir()):
        if not species_directory.is_dir():
            continue
        species_id = species_directory.name
        report["species"][species_id] = {"total": 0, "valid": 0, "invalid": 0, "files": []}

        for audio_path in species_directory.rglob("*"):
            if not audio_path.is_file():
                continue
            report["summary"]["total"] += 1
            report["species"][species_id]["total"] += 1

            extra = {}
            if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
                valid, reason = False, "unsupported_format"
            else:
                file_hash = sha256_file(audio_path)
                if file_hash in seen_hashes:
                    valid, reason = False, "duplicate"
                    report["summary"]["duplicates"] += 1
                else:
                    seen_hashes[file_hash] = str(audio_path)
                    valid, reason, extra = validate_audio(audio_path)
                    if valid:
                        duration = extra["duration"]
                        if duration < MIN_AUDIO_DURATION_SECONDS:
                            valid, reason = False, "too_short"
                        elif duration > MAX_AUDIO_DURATION_SECONDS:
                            valid, reason = False, "too_long"

            report["species"][species_id]["files"].append({"path": str(audio_path), "valid": valid, "reason": reason, **extra})
            if valid:
                report["summary"]["valid"] += 1
                report["species"][species_id]["valid"] += 1
            else:
                report["summary"]["invalid"] += 1
                report["species"][species_id]["invalid"] += 1

    output_path = PROJECT_ROOT / "metadata" / "validation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(json.dumps(report["summary"], indent=2))
    print("\nPER-SPECIES SUMMARY\n" + "-" * 50)
    for species_id, data in report["species"].items():
        print(f"{species_id}: {data['valid']} valid / {data['total']} total")
    print(f"\nReport saved to: {output_path}")

if __name__ == "__main__":
    main()
