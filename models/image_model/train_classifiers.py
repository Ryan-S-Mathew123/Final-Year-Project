"""Train the shared classifier families over ConvNeXt image embeddings."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.image_model.feature_extractor import IMAGE_TRANSFORM, load_image_encoder
from models.model_families import MODEL_ARTIFACTS, train_comparison_models


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "convnext_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32


def extract_embeddings(encoder: torch.nn.Module, split_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dataset = datasets.ImageFolder(split_dir, transform=IMAGE_TRANSFORM)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    features = []
    labels = []
    with torch.inference_mode():
        for images, targets in loader:
            embeddings = encoder(images.to(DEVICE))
            features.append(embeddings.cpu().numpy())
            labels.append(targets.numpy())
    if not features:
        raise ValueError(f"No images found in split: {split_dir}")
    targets = np.concatenate(labels)
    return np.concatenate(features), np.asarray(dataset.classes)[targets], dataset.classes


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing ConvNeXt feature checkpoint: {CHECKPOINT_PATH}. "
            "Train the image model first with models/image_model/train.py."
        )

    encoder, checkpoint_classes = load_image_encoder(CHECKPOINT_PATH, DEVICE)
    splits = {}
    for split_name in ("train", "val", "test"):
        split_path = BASE_DIR / "dataset" / split_name
        print(f"Extracting {split_name} image embeddings...", flush=True)
        features, labels, classes = extract_embeddings(encoder, split_path)
        if classes != checkpoint_classes:
            raise ValueError(f"{split_name} classes do not match the ConvNeXt checkpoint.")
        splits[split_name] = features, labels
        print(f"{split_name}: {len(labels)} images, {len(classes)} classes")

    train_comparison_models(
        *splits["train"],
        *splits["val"],
        *splits["test"],
        checkpoint_classes,
        BASE_DIR / "comparison_models",
        "image",
    )
    print(f"Saved {len(MODEL_ARTIFACTS)} model families and holdout metrics.")


if __name__ == "__main__":
    main()
