# Bird Image Classification Model

This module implements the image-based bird species classification component of the Final Year Project.

The objective is to classify bird images into 12 different bird species using deep learning and compare multiple pretrained CNN architectures.

The following four pretrained ImageNet models were fine-tuned and evaluated:

1. ResNet50
2. EfficientNet-B0
3. MobileNetV3-Large
4. ConvNeXt-Tiny

The models were trained using transfer learning and evaluated on a separate held-out test set.

---

## Project Overview

This image classification module is designed to identify bird species from images.

The workflow is:

Dataset
    ↓
Train / Validation / Test Split
    ↓
Image Augmentation
    ↓
Transfer Learning
    ↓
Four CNN Architectures
    ↓
Model Evaluation
    ↓
Performance Comparison

---

## Dataset

The dataset contains **10,807 bird images** belonging to 12 species.

### Bird Species

| Species | Images |
|---|---:|
| Asian Koel | 1,000 |
| Common Cuckoo | 1,000 |
| Grey Junglefowl | 728 |
| Indian Grey Hornbill | 1,000 |
| Indian Peafowl | 1,000 |
| Indian Robin | 1,000 |
| Peregrine Falcon | 1,000 |
| Rock Eagle Owl | 655 |
| Rose-ringed Parakeet | 1,000 |
| Spotted Dove | 1,000 |
| Spotted Owlet | 1,000 |
| White-naped Woodpecker | 424 |
| **Total** | **10,807** |

The dataset contains class imbalance because some species have fewer available images.

The original dataset is preserved in:

```text
dataset/raw/
```

---

## Dataset Split

The dataset was divided using a fixed random seed of 42.

| Split | Percentage | Images |
|---|---:|---:|
| Training | 70% | 7,563 |
| Validation | 15% | 1,620 |
| Testing | 15% | 1,624 |
| **Total** | **100%** | **10,807** |

The original images in `dataset/raw/` were not modified.

---

## Training Classes

The training split contains the following number of images per class:

| Species | Training Images |
|---|---:|
| Asian Koel | 700 |
| Common Cuckoo | 700 |
| Grey Junglefowl | 509 |
| Indian Grey Hornbill | 700 |
| Indian Peafowl | 700 |
| Indian Robin | 700 |
| Peregrine Falcon | 700 |
| Rock Eagle Owl | 458 |
| Rose-ringed Parakeet | 700 |
| Spotted Dove | 700 |
| Spotted Owlet | 700 |
| White-naped Woodpecker | 296 |

Class-weighted Cross Entropy Loss was used during training to reduce the effect of class imbalance.

---

# Models

Four pretrained architectures were fine-tuned using ImageNet pretrained weights.

## ResNet50

A residual convolutional neural network using skip connections to improve the training of deeper networks.

## EfficientNet-B0

A lightweight architecture designed to achieve a balance between accuracy and computational efficiency.

## MobileNetV3-Large

A lightweight architecture designed primarily for efficient inference and deployment on resource-constrained devices.

## ConvNeXt-Tiny

A modern convolutional architecture based on design principles inspired by newer vision architectures while retaining a convolutional backbone.

---

# Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 20 |
| Optimizer | AdamW |
| Initial Learning Rate | 0.0001 |
| Weight Decay | 0.0001 |
| Learning Rate Scheduler | ReduceLROnPlateau |
| Scheduler Factor | 0.5 |
| Scheduler Patience | 2 |
| Loss Function | Class-weighted Cross Entropy |
| Pretrained Weights | ImageNet |
| Random Seed | 42 |
| Device | NVIDIA CUDA GPU |

### Training Augmentation

The training pipeline used:

- Random Resized Crop
- Random Horizontal Flip
- Color Jitter
- Random Rotation
- ImageNet normalization

### Validation/Test Preprocessing

Validation and test images used:

- Resize to 256
- Center Crop to 224 × 224
- ImageNet normalization

---

# Validation Results

The best validation accuracy obtained during training was:

| Model | Best Validation Accuracy |
|---|---:|
| ResNet50 | 90.43% |
| EfficientNet-B0 | 88.77% |
| MobileNetV3-Large | 86.23% |
| ConvNeXt-Tiny | 93.09% |

These values were obtained from the validation set during the 20-epoch training process.

---

# Final Test Results

After training, the best checkpoint for each architecture was evaluated on the independent test set.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| ResNet50 | 91.07% | 91.24% | 91.07% | 91.07% |
| EfficientNet-B0 | 89.22% | 89.30% | 89.22% | 89.24% |
| MobileNetV3-Large | 85.65% | 86.06% | 85.65% | 85.71% |
| ConvNeXt-Tiny | 92.86% | 92.98% | 92.86% | 92.88% |

All metrics were calculated on the same held-out test set.

---

# Evaluation

The evaluation script calculates:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- Macro Precision
- Macro Recall
- Macro F1 Score
- Number of parameters
- Model checkpoint size
- Average inference time
- Images processed per second
- Classification report
- Confusion matrix

The evaluation results are stored in:

```text
results/
```

### Generated Result Files

```text
results/
├── convnext_history.json
├── convnext_test_results.json
├── efficientnet_history.json
├── efficientnet_test_results.json
├── mobilenet_history.json
├── mobilenet_test_results.json
├── model_comparison.json
├── resnet50_history.json
└── resnet50_test_results.json
```

---

# Project Structure

```text
bird_image_model/
│
├── README.md
│
├── train.py
├── evaluate.py
├── split_dataset.py
│
├── checkpoints/
│   ├── convnext_best.pth
│   ├── efficientnet_best.pth
│   ├── mobilenet_best.pth
│   └── resnet50_best.pth
│
├── dataset/
│   ├── raw/
│   │   ├── asian_koel/
│   │   ├── common_cuckoo/
│   │   ├── grey_junglefowl/
│   │   ├── indian_grey_hornbill/
│   │   ├── indian_peafowl/
│   │   ├── indian_robin/
│   │   ├── peregrine_falcon/
│   │   ├── rock_eagle_owl/
│   │   ├── rose_ringed_parakeet/
│   │   ├── spotted_dove/
│   │   ├── spotted_owlet/
│   │   └── white_naped_woodpecker/
│   │
│   ├── train/
│   ├── val/
│   └── test/
│
└── results/
    ├── training histories
    ├── test results
    └── model comparison
```

---

# Requirements

The model was developed using Python 3.11 and PyTorch with CUDA support.

Main dependencies:

```text
Python 3.11
PyTorch
TorchVision
NumPy
Pandas
Scikit-learn
Pillow
Matplotlib
Seaborn
tqdm
```

---

# Training

From the `bird_image_model` directory:

```bash
python split_dataset.py
```

Train an individual model:

```bash
python train.py --model resnet50
```

```bash
python train.py --model efficientnet
```

```bash
python train.py --model mobilenet
```

```bash
python train.py --model convnext
```

The best validation checkpoint is automatically saved in:

```text
checkpoints/
```

---

# Evaluation

To evaluate all four trained models:

```bash
python evaluate.py
```

The evaluation script automatically loads:

```text
resnet50_best.pth
efficientnet_best.pth
mobilenet_best.pth
convnext_best.pth
```

and evaluates them on:

```text
dataset/test/
```

The final comparison is saved as:

```text
results/model_comparison.json
```

---

# Experimental Result

The experiment demonstrates that the four architectures produce different classification performance on the same bird image dataset.

The final test-set results were:

- ResNet50: 91.07% accuracy
- EfficientNet-B0: 89.22% accuracy
- MobileNetV3-Large: 85.65% accuracy
- ConvNeXt-Tiny: 92.86% accuracy

The results provide a quantitative comparison of the four architectures for the bird image classification task.

---

# Future Integration

The trained image classification module is intended to form one part of the larger Final Year Project.

The overall system is planned to combine:

```text
Bird Image
     ↓
Image Classification Model
     ↓
Bird Species + Confidence

Bird Audio
     ↓
Audio Classification Model
     ↓
Bird Species + Confidence

        ↓
 Multimodal Fusion
        ↓
Final Bird Prediction
```

The image model can subsequently be exposed through the project server and integrated with the audio-based bird localization pipeline.

---

# Authors

Final Year Project

B.Tech Computer Science and Engineering (AI/ML)

Christ University, Bangalore
