import os
import json
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = [
    "resnet50",
    "efficientnet",
    "mobilenet",
    "convnext"
]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(model_name, num_classes):

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    elif model_name == "mobilenet":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            num_classes
        )

    elif model_name == "convnext":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features,
            num_classes
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


# ============================================================
# TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model_name):

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"{model_name}_best.pth"
    )

    if not os.path.exists(checkpoint_path):
        print(f"\nCheckpoint not found: {checkpoint_path}")
        return None

    print("\n" + "=" * 70)
    print(f"Evaluating: {model_name}")
    print("=" * 70)

    # Dataset
    dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    classes = dataset.classes
    num_classes = len(classes)

    # Model
    model = create_model(model_name, num_classes)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    print(f"Device: {DEVICE}")
    print(f"Test images: {len(dataset)}")
    print(f"Classes: {num_classes}")

    # Parameters
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # Predictions
    all_predictions = []
    all_targets = []

    total_inference_time = 0.0

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            outputs = model(images)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            total_inference_time += end - start

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                labels.cpu().numpy()
            )

    # Metrics
    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    precision_weighted = precision_score(
        all_targets,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        all_targets,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        all_targets,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    precision_macro = precision_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    # Classification report
    report = classification_report(
        all_targets,
        all_predictions,
        target_names=classes,
        zero_division=0,
        output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(
        all_targets,
        all_predictions
    )

    # Inference speed
    total_images = len(all_targets)

    avg_time_per_image = (
        total_inference_time / total_images
    )

    images_per_second = (
        total_images / total_inference_time
        if total_inference_time > 0
        else 0
    )

    # Checkpoint size
    checkpoint_size_mb = (
        os.path.getsize(checkpoint_path)
        / (1024 * 1024)
    )

    # Results
    results = {
        "model": model_name,
        "test_images": total_images,
        "classes": classes,

        "accuracy": accuracy,

        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,

        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,

        "total_parameters": total_params,
        "trainable_parameters": trainable_params,

        "checkpoint_size_mb": checkpoint_size_mb,

        "total_inference_time_seconds":
            total_inference_time,

        "average_inference_time_ms":
            avg_time_per_image * 1000,

        "images_per_second":
            images_per_second,

        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    # Save individual result
    output_path = os.path.join(
        RESULTS_DIR,
        f"{model_name}_test_results.json"
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print results
    print("\nRESULTS")
    print("-" * 50)

    print(f"Accuracy:          {accuracy * 100:.2f}%")
    print(f"Precision:         {precision_weighted * 100:.2f}%")
    print(f"Recall:            {recall_weighted * 100:.2f}%")
    print(f"F1 Score:          {f1_weighted * 100:.2f}%")

    print("\nMacro Metrics")
    print(f"Macro Precision:   {precision_macro * 100:.2f}%")
    print(f"Macro Recall:      {recall_macro * 100:.2f}%")
    print(f"Macro F1:          {f1_macro * 100:.2f}%")

    print("\nModel Information")
    print(f"Parameters:        {total_params:,}")
    print(f"Checkpoint size:   {checkpoint_size_mb:.2f} MB")

    print("\nInference")
    print(f"Total time:        {total_inference_time:.2f} sec")
    print(f"Per image:         {avg_time_per_image * 1000:.2f} ms")
    print(f"Images/sec:        {images_per_second:.2f}")

    print(f"\nSaved: {output_path}")

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BIRD IMAGE MODEL - TEST EVALUATION")
    print("=" * 70)

    print(f"\nTest directory: {TEST_DIR}")
    print(f"Device: {DEVICE}")

    all_results = {}

    for model_name in MODELS:

        result = evaluate_model(model_name)

        if result is not None:
            all_results[model_name] = result

    # Save comparison
    comparison = []

    for model_name, result in all_results.items():

        comparison.append({
            "model": model_name,
            "accuracy": result["accuracy"],
            "precision_weighted":
                result["precision_weighted"],
            "recall_weighted":
                result["recall_weighted"],
            "f1_weighted":
                result["f1_weighted"],
            "precision_macro":
                result["precision_macro"],
            "recall_macro":
                result["recall_macro"],
            "f1_macro":
                result["f1_macro"],
            "parameters":
                result["total_parameters"],
            "checkpoint_size_mb":
                result["checkpoint_size_mb"],
            "inference_ms":
                result["average_inference_time_ms"],
            "images_per_second":
                result["images_per_second"]
        })

    comparison_path = os.path.join(
        RESULTS_DIR,
        "model_comparison.json"
    )

    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=4)

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        f"{'Model':<18}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    print("-" * 66)

    for result in comparison:

        print(
            f"{result['model']:<18}"
            f"{result['accuracy'] * 100:<12.2f}"
            f"{result['precision_weighted'] * 100:<12.2f}"
            f"{result['recall_weighted'] * 100:<12.2f}"
            f"{result['f1_weighted'] * 100:<12.2f}"
        )

    print(f"\nComparison saved to:")
    print(comparison_path)


if __name__ == "__main__":
    main()