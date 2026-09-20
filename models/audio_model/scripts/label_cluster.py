"""
Label or discard a low-confidence cluster.

Usage:
    python scripts/label_cluster.py --cluster 3 --species indian_robin
    python scripts/label_cluster.py --cluster 5 --discard
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import ensure_directories
from src.audio.low_confidence_collector import LowConfidenceCollector


def main() -> None:
    parser = argparse.ArgumentParser(description="Label or discard a low-confidence cluster")
    parser.add_argument("--cluster", type=int, required=True, help="Cluster ID to label or discard")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--species", type=str, help="Species ID to assign (e.g. indian_robin)")
    group.add_argument("--discard", action="store_true", help="Discard the cluster (noise / non-bird)")
    args = parser.parse_args()

    ensure_directories()
    collector = LowConfidenceCollector()

    if args.discard:
        count = collector.discard_cluster(args.cluster)
        if count:
            print(f"\n✓ Discarded {count} samples from cluster {args.cluster}")
        else:
            print(f"\n✗ No unlabelled samples found in cluster {args.cluster}")
    else:
        species_id = args.species.strip().lower().replace("-", "_").replace(" ", "_")
        count = collector.label_cluster(args.cluster, species_id)
        if count:
            print(f"\n✓ Promoted {count} samples from cluster {args.cluster} → '{species_id}'")
            print(f"  Audio files moved to: datasets/audio/{species_id}/")
            print(f"  Embeddings regenerated and prototypes rebuilt.")
            print(f"\n  Next step: retrain the model:")
            print(f"    python train.py")
        else:
            print(f"\n✗ No unlabelled samples found in cluster {args.cluster}")


if __name__ == "__main__":
    main()
