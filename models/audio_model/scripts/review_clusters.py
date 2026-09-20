"""
Review low-confidence clusters.

Usage:
    python scripts/review_clusters.py              # show existing clusters
    python scripts/review_clusters.py --recluster  # re-run clustering first
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
    parser = argparse.ArgumentParser(description="Review low-confidence bird audio clusters")
    parser.add_argument(
        "--recluster", action="store_true",
        help="Re-run agglomerative clustering before displaying results",
    )
    parser.add_argument(
        "--distance", type=float, default=None,
        help="Override the clustering distance threshold (default from config)",
    )
    args = parser.parse_args()

    ensure_directories()
    collector = LowConfidenceCollector()

    pending = collector.count()
    print(f"\n{'=' * 70}")
    print(f"LOW-CONFIDENCE RECORDINGS: {pending} unlabelled samples")
    print(f"{'=' * 70}")

    if pending == 0:
        print("No low-confidence recordings to review.")
        return

    if args.recluster:
        print("\nRe-clustering...")
        kwargs = {}
        if args.distance is not None:
            kwargs["distance_threshold"] = args.distance
        result = collector.cluster(**kwargs)
        print(f"  → {result['n_clusters']} clusters found")
        print(f"  → Cluster sizes: {result['cluster_sizes']}")

    summary = collector.get_clusters_summary()

    if not summary:
        print("\nNo clusters assigned yet. Run with --recluster to cluster.")
        return

    print(f"\n{'─' * 70}")
    print(f"{'Cluster':>8}  {'Size':>5}  {'Avg Similarity':>15}  Sample Files")
    print(f"{'─' * 70}")

    for cluster_id, info in sorted(summary.items()):
        sample_names = [Path(s["saved_path"]).name for s in info["samples"][:3]]
        more = f"  (+{info['count'] - 3} more)" if info["count"] > 3 else ""
        print(
            f"  {cluster_id:>6}  {info['count']:>5}  "
            f"{info['avg_similarity']:>15.4f}  "
            f"{', '.join(sample_names)}{more}"
        )

    print(f"{'─' * 70}")
    print(f"\nTo label a cluster:")
    print(f"  python scripts/label_cluster.py --cluster <ID> --species <species_id>")
    print(f"\nTo discard a noise cluster:")
    print(f"  python scripts/label_cluster.py --cluster <ID> --discard")
    print()


if __name__ == "__main__":
    main()
