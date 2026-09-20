import random
import shutil
from pathlib import Path

# =========================
# Configuration
# =========================

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "dataset" / "raw"
OUTPUT_DIR = BASE_DIR / "dataset"

random.seed(SEED)


# =========================
# Helper functions
# =========================

def get_images(folder):
    """Return all supported image files in a folder."""
    return [
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def split_images(images):
    """Split images into train, validation and test sets."""
    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    return train_images, val_images, test_images


def copy_images(images, destination):
    """Copy images to destination folder."""
    destination.mkdir(parents=True, exist_ok=True)

    for image in images:
        shutil.copy2(image, destination / image.name)


# =========================
# Main
# =========================

def main():

    if not RAW_DIR.exists():
        print(f"ERROR: Raw dataset not found:")
        print(RAW_DIR)
        return

    # Find class folders
    class_folders = sorted(
        folder for folder in RAW_DIR.iterdir()
        if folder.is_dir()
    )

    if not class_folders:
        print("ERROR: No class folders found inside dataset/raw/")
        return

    print("=" * 60)
    print("BIRD IMAGE DATASET SPLITTER")
    print("=" * 60)
    print(f"Random seed : {SEED}")
    print(f"Train       : {TRAIN_RATIO * 100:.0f}%")
    print(f"Validation  : {VAL_RATIO * 100:.0f}%")
    print(f"Test        : {TEST_RATIO * 100:.0f}%")
    print(f"Classes     : {len(class_folders)}")
    print("=" * 60)

    total_images = 0

    for class_folder in class_folders:

        class_name = class_folder.name

        images = get_images(class_folder)

        if not images:
            print(f"\nWARNING: {class_name} contains no images.")
            continue

        train_images, val_images, test_images = split_images(images)

        # Destination folders
        train_dir = OUTPUT_DIR / "train" / class_name
        val_dir = OUTPUT_DIR / "val" / class_name
        test_dir = OUTPUT_DIR / "test" / class_name

        # Copy files
        copy_images(train_images, train_dir)
        copy_images(val_images, val_dir)
        copy_images(test_images, test_dir)

        total = len(images)
        total_images += total

        print(
            f"{class_name:<28}"
            f"Total: {total:<5}"
            f"Train: {len(train_images):<5}"
            f"Val: {len(val_images):<5}"
            f"Test: {len(test_images):<5}"
        )

    print("=" * 60)
    print(f"TOTAL IMAGES: {total_images}")
    print("=" * 60)

    print("\nDataset split completed successfully!")

    print("\nOutput structure:")
    print("dataset/")
    print("├── raw/")
    print("├── train/")
    print("├── val/")
    print("└── test/")


if __name__ == "__main__":
    main()