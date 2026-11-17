"""ConvNeXt feature extraction shared by image training and inference."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms


IMAGE_SIZE = 224
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_image_encoder(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    classes = list(checkpoint["classes"])
    encoder = models.convnext_tiny(weights=None)
    encoder.classifier[2] = nn.Linear(encoder.classifier[2].in_features, len(classes))
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.classifier[2] = nn.Identity()
    encoder.to(device).eval()
    return encoder, classes
