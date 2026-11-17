"""Train all shared classifier families for the audio modality."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.model_families import train_comparison_models


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset"
SEED = 42
MODEL_CHOICES = {
    "random_forest": "Random Forest",
    "svm": "SVM",
    "knn": "K-Nearest Neighbors",
    "gradient_boosting": "Gradient Boosting",
}


def extract_features(file_path: Path) -> np.ndarray | None:
    try:
        audio, sample_rate = librosa.load(file_path, mono=True)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as exc:
        print(f"Error parsing {file_path}: {exc}")
        return None


def process_directory(dir_path: Path) -> tuple[list[np.ndarray], list[str]]:
    features = []
    labels = []
    for class_dir in sorted(dir_path.iterdir()):
        if not class_dir.is_dir():
            continue
        for file_path in sorted(class_dir.iterdir()):
            if file_path.suffix.lower() not in {".mp3", ".wav", ".ogg", ".flac", ".m4a"}:
                continue
            data = extract_features(file_path)
            if data is not None:
                features.append(data)
                labels.append(class_dir.name)
    return features, labels


def split_dataset(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, ...]:
    """Make deterministic, stratified 70/15/15 train/validation/test splits."""
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        features,
        labels,
        test_size=0.15,
        random_state=SEED,
        stratify=labels,
    )
    validation_fraction = 0.15 / 0.85
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=validation_fraction,
        random_state=SEED,
        stratify=y_train_validation,
    )
    return X_train, y_train, X_validation, y_validation, X_test, y_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Train comparable audio classifiers.")
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", *MODEL_CHOICES],
        help="Train every shared family, or just one family.",
    )
    args = parser.parse_args()

    if not DATASET_PATH.is_dir():
        raise FileNotFoundError(f"Audio dataset directory not found: {DATASET_PATH}")
    features, labels = process_directory(DATASET_PATH)
    if not features:
        raise ValueError(f"No supported audio files found under {DATASET_PATH}")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    model_names = None
    if args.model != "all":
        model_names = [MODEL_CHOICES[args.model]]

    splits = split_dataset(X, y)
    print(f"Training on {len(splits[0])} audio clips across {len(np.unique(y))} classes.")
    train_comparison_models(
        *splits,
        sorted(np.unique(y).tolist()),
        BASE_DIR / "comparison_models",
        "audio",
        model_names,
    )
    print(f"Saved audio classifier artifacts under {BASE_DIR / 'comparison_models'}.")
    print(f"Holdout comparison saved to {BASE_DIR / 'results' / 'classifier_comparison.json'}.")


if __name__ == "__main__":
    main()
