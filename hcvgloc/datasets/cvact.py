"""
cvact.py
---------
CVACT dataset loader for HCVGLoc Stage 1.

CVACT_val is the standard cross-area evaluation benchmark — it contains
locations UNSEEN during training, making it the gold standard for measuring
cross-area generalisation.

Dataset layout:
    <root>/
        ACT_data/          # satellite crops (512×512)
            <uuid>.jpg
        ACT_queries_real/  # ground panoramas (query)
            <uuid>.jpg
        ACT_data_split.csv # uuid, lat, lon, split(train/val/test)

Reference:
    Liu & Li, "Lending Orientation to Neural Networks for Cross-view Geo-localization"
    CVPR 2019.
"""

import csv
import json
from pathlib import Path
from typing import Optional, Callable

from hcvgloc.datasets.base_dataset import CrossViewDataset, GalleryDataset
from hcvgloc.datasets.transforms import val_transform


class CVACTDataset(CrossViewDataset):
    """
    CVACT cross-view dataset.

    Splits:
        'train'  — ~73,185 pairs (same-area)
        'val'    — ~35,532 pairs (cross-area — HELD OUT, primary benchmark)
    """

    DOMAIN_ID = CrossViewDataset.DOMAIN_IDS["cvact"]

    def __init__(
        self,
        root: str,
        split: str = "val",
        query_transform: Optional[Callable] = None,
        sat_transform: Optional[Callable] = None,
    ):
        super().__init__(root, split, query_transform, sat_transform)

    def _load_pairs(self):
        csv_path = self.root / "ACT_data_split.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CVACT split file not found: {csv_path}")

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if row["split"].strip() != self.split:
                    continue
                uuid = row["uuid"].strip()
                query_path = str(self.root / "ACT_queries_real" / f"{uuid}.jpg")
                sat_path   = str(self.root / "ACT_data"         / f"{uuid}.jpg")
                self.pairs.append((query_path, sat_path, idx, self.DOMAIN_ID))


class CVACTGallery(GalleryDataset):
    """Satellite-only gallery for CVACT Recall@K evaluation."""

    def __init__(self, root: str, split: str = "val", transform=None):
        if transform is None:
            transform = val_transform()

        root = Path(root)
        csv_path = root / "ACT_data_split.csv"
        sat_paths, labels = [], []
        with open(csv_path, "r") as f:
            for idx, row in enumerate(csv.DictReader(f)):
                if row["split"].strip() != split:
                    continue
                uuid = row["uuid"].strip()
                sat_paths.append(str(root / "ACT_data" / f"{uuid}.jpg"))
                labels.append(idx)
        super().__init__(sat_paths, labels, transform)
