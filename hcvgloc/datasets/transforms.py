"""
transforms.py
--------------
Augmentation pipelines for HCVGLoc Stage 1 training.

Two separate pipelines:
    query_train_transform   — aggressive augmentation for UAV query images
    sat_train_transform     — lighter augmentation for satellite images
    val_transform           — deterministic resize + normalise (both views)

Key design decisions:
    - 384×384 input (matches EfficientViT-B1 training resolution)
    - Satellite images: no horizontal flip (north-up convention)
    - Query images: random rotation to encourage rotation invariance
    - Both: colour jitter to handle cross-season / cross-time-of-day changes
"""

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


# ImageNet statistics (both query and satellite images are natural images)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 384


def query_train_transform() -> T.Compose:
    """
    Aggressive augmentation for UAV query (ground-facing) images.
    - Random crop simulates varying altitude / FOV
    - Random rotation handles UAV yaw variance
    - Colour jitter handles lighting changes
    """
    return T.Compose([
        T.Resize((IMG_SIZE + 32, IMG_SIZE + 32), interpolation=InterpolationMode.BICUBIC),
        T.RandomCrop(IMG_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def sat_train_transform() -> T.Compose:
    """
    Lighter augmentation for satellite reference images.
    - NO horizontal flip (would change cardinal direction)
    - NO rotation (satellite images are north-up by convention)
    - Mild colour jitter for seasonal variation
    """
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def val_transform() -> T.Compose:
    """
    Deterministic transform for validation / inference (both views).
    """
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
