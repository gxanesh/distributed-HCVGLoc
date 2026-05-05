"""
cvusa.py
---------
CVUSA dataset loader for HCVGLoc Stage 1.

Dataset layout expected on disk:
    <root>/
        streetview/       # query panoramas (ground-level)
            0000001.jpg
            ...
        bingmap/          # satellite crops
            0000001.png
            ...
        splits/
            train-19zl.csv   # CSV: sat_path, pano_path, lat, lon
            val-19zl.csv

Each row in the CSV:
    bingmap/0000001.png,streetview/0000001.jpg,37.123,-122.456

Dataset stats (standard split):
    Train: ~35,532 pairs  |  Val: ~8,884 pairs
    Gallery: ~35,532 unique satellite images (train)
             ~8,884 unique satellite images (val)

Download:
    https://github.com/viibridges/crossnet  (original)
    or via rclone from lab storage
"""

import csv
from pathlib import Path
from typing import Optional, Callable

from hcvgloc.datasets.base_dataset import CrossViewDataset, GalleryDataset
from hcvgloc.datasets.transforms import val_transform


class CVUSADataset(CrossViewDataset):
    """
    CVUSA cross-view geo-localization dataset.

    Args:
        root:              path to CVUSA root directory
        split:             'train' or 'val'
        query_transform:   transform applied to panorama (query) images
        sat_transform:     transform applied to satellite images
    """

    DOMAIN_ID = CrossViewDataset.DOMAIN_IDS["cvusa"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        query_transform: Optional[Callable] = None,
        sat_transform: Optional[Callable] = None,
    ):
        super().__init__(root, split, query_transform, sat_transform)

    def _load_pairs(self):
        split_file = {
            "train": "splits/train-19zl.csv",
            "val":   "splits/val-19zl.csv",
        }.get(self.split)

        if split_file is None:
            raise ValueError(f"Unknown split '{self.split}'. Use 'train' or 'val'.")

        csv_path = self.root / split_file
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CVUSA split file not found: {csv_path}\n"
                "Please download the dataset and place it under the expected structure."
            )

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                sat_rel, query_rel = row[0].strip(), row[1].strip()
                sat_path   = str(self.root / sat_rel)
                query_path = str(self.root / query_rel)
                self.pairs.append((query_path, sat_path, idx, self.DOMAIN_ID))


class CVUSAGallery(GalleryDataset):
    """
    Satellite-only gallery for CVUSA Recall@K evaluation.

    Deduplicates satellite images (each unique sat tile appears once).
    """

    def __init__(self, root: str, split: str = "val", transform=None):
        if transform is None:
            transform = val_transform()

        root = Path(root)
        split_file = {
            "train": "splits/train-19zl.csv",
            "val":   "splits/val-19zl.csv",
        }[split]

        csv_path = root / split_file
        seen = {}
        with open(csv_path, "r") as f:
            for idx, row in enumerate(csv.reader(f)):
                sat_rel = row[0].strip()
                sat_path = str(root / sat_rel)
                if sat_path not in seen:
                    seen[sat_path] = idx

        sat_paths = list(seen.keys())
        labels = list(seen.values())
        super().__init__(sat_paths, labels, transform)
