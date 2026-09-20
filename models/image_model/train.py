import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "val"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Transforms
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.7, 1.0)
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# Model Factory
# ============================================================

def create_model(model_name, num_classes):

    if model_name == "resnet50":

        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            num_classes
        )

    elif model_name == "efficientnet":

        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    elif model_name == "mobilenet":

        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)

        in_features = model.classifier[3].in_features

        model.classifier[3] = nn.Linear(
            in_features,
            num_classes
        )

    elif model_name == "convnext":

        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)

        in_features = model.classifier[2].in_features

        model.classifier[2] = nn.Linear(
            in_features,
            num_classes
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


# ============================================================
# Class Weights
# ============================================================

def calculate_class_weights(dataset):

    targets = np.array(dataset.targets)

    class_counts = np.bincount(
        targets,
        minlength=len(dataset.classes)
    )

    total = len(targets)

    weights = total / (
        len(dataset.classes) * class_counts
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32
    )

    return weights, class_counts


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Validation
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "resnet50",
            "efficientnet",
            "mobilenet",
            "convnext"
        ]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS
    )

    args = parser.parse_args()

    model_name = args.model

    print("=" * 70)
    print("BIRD IMAGE CLASSIFICATION")
    print("=" * 70)

    print(f"Model       : {model_name}")
    print(f"Device      : {DEVICE}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Image size  : {IMAGE_SIZE}")
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=val_transform
    )

    print(f"\nClasses: {len(train_dataset.classes)}")

    for index, class_name in enumerate(
        train_dataset.classes
    ):
        print(
            f"{index:2d} -> {class_name}"
        )

    print(
        f"\nTraining images   : {len(train_dataset)}"
    )

    print(
        f"Validation images : {len(val_dataset)}"
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights, class_counts = (
        calculate_class_weights(
            train_dataset
        )
    )

    class_weights = class_weights.to(
        DEVICE
    )

    print("\nTraining class counts:")

    for class_name, count in zip(
        train_dataset.classes,
        class_counts
    ):
        print(
            f"{class_name:<28} {count}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        model_name,
        len(train_dataset.classes)
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_accuracy = 0.0
    best_model_state = None

    history = []

    start_time = time.time()

    for epoch in range(args.epochs):

        epoch_start = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        print(
            f"\nEpoch [{epoch + 1}/{args.epochs}]"
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Acc:   {val_accuracy:.4f}"
        )

        print(
            f"Learning Rate: {current_lr:.7f}"
        )

        print(
            f"Time: {epoch_time:.1f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr
        })

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            CHECKPOINT_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            checkpoint_path = (
                CHECKPOINT_DIR /
                f"{model_name}_best.pth"
            )

            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": best_model_state,
                    "classes": train_dataset.classes,
                    "best_val_accuracy": best_val_accuracy,
                },
                checkpoint_path
            )

            print(
                f"✓ Best model saved: {checkpoint_path}"
            )

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

    total_time = time.time() - start_time

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    history_path = (
        RESULTS_DIR /
        f"{model_name}_history.json"
    )

    with open(
        history_path,
        "w"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Model: {model_name}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Training time: "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        f"Checkpoint: "
        f"checkpoints/{model_name}_best.pth"
    )

    print(
        f"History: "
        f"results/{model_name}_history.json"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()