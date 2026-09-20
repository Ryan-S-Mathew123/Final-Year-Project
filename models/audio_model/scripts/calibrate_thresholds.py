from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.metadata_manager import MetadataManager

def main() -> None:
    records = MetadataManager().samples_with_embeddings()
    if len(records) < 3:
        raise RuntimeError("Not enough embeddings for calibration.")

    embeddings = np.vstack([np.load(record["embedding_path"]) for record in records])
    labels = [record["species_id"] for record in records]
    positive_scores, negative_scores = [], []

    for i in range(len(records)):
        similarities = embeddings @ embeddings[i]
        for j in range(len(records)):
            if i == j:
                continue
            if labels[i] == labels[j]:
                positive_scores.append(float(similarities[j]))
            else:
                negative_scores.append(float(similarities[j]))

    if not positive_scores or not negative_scores:
        raise RuntimeError("Need multiple species and multiple samples.")

    positive_lower = np.quantile(positive_scores, 0.05)
    negative_upper = np.quantile(negative_scores, 0.95)
    suggested_threshold = (positive_lower + negative_upper) / 2

    print("THRESHOLD CALIBRATION")
    print("-" * 40)
    print(f"5th percentile positive similarity: {positive_lower:.4f}")
    print(f"95th percentile negative similarity: {negative_upper:.4f}")
    print(f"\nSuggested threshold: {suggested_threshold:.4f}")

if __name__ == "__main__":
    main()
