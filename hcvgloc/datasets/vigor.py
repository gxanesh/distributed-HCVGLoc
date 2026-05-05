"""
vigor.py
---------
VIGOR dataset loader for HCVGLoc Stage 1.

VIGOR is the most challenging benchmark because:
  - Multi-positive: each query has ~4 matching satellite tiles
  - Same-area and cross-area splits
  - Urban scenes with repeated structures

Dataset layout:
    <root>/
        satellite/                    # 90,618 satellite tiles
            <city>/<tile_id>.png
        panorama/                     # 105,214 panoramas
            <city>/<pano_id>.jpg
        splits/
            same-area/
                train.json            # {pano_id: [sat_id, ...], ...}
                val.json
            cross-area/
                train.json
                val.json

Reference:
    VIGOR: Cross-View Geo-localization beyond One-to-one Retrieval
    (Zhu et al., CVPR 2021)
"""

import json
from pathlib import Path
from typing import Optional, Callable, List

import torch
from torch.utils.data import Dataset
from PIL import Image

from hcvgloc.datasets.base_dataset import CrossViewDataset, GalleryDataset
from hcvgloc.datasets.transforms import val_transform


class VIGORDataset(Dataset):
    """
    VIGOR cross-view dataset with multi-positive support.

    Each sample returns a query image paired with ONE positive satellite
    (randomly chosen from its positive set during training).
    The full positive set is returned for evaluation.

    Args:
        root:       path to VIGOR root
        split:      'train' or 'val'
        area:       'same' (same-area) or 'cross' (cross-area)
        query_transform, sat_transform: torchvision transforms
    """

    DOMAIN_ID = CrossViewDataset.DOMAIN_IDS["vigor"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        area: str = "same",
        query_transform: Optional[Callable] = None,
        sat_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.area = area
        self.query_transform = query_transform
        self.sat_transform = sat_transform

        self.queries: List[dict] = []   # [{pano_path, sat_paths[], label}]
        self._load_pairs()

    def _load_pairs(self):
        json_path = self.root / "splits" / f"{self.area}-area" / f"{self.split}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"VIGOR split file not found: {json_path}\n"
                "Expected structure: splits/same-area/train.json"
            )

        with open(json_path) as f:
            data = json.load(f)

        for idx, (pano_id, sat_ids) in enumerate(data.items()):
            pano_path = str(self.root / "panorama" / pano_id)
            sat_paths = [str(self.root / "satellite" / sid) for sid in sat_ids]
            self.queries.append({
                "pano_path": pano_path,
                "sat_paths": sat_paths,
                "label": idx,
            })

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict:
        item = self.queries[idx]

        query_img = Image.open(item["pano_path"]).convert("RGB")
        if self.query_transform:
            query_img = self.query_transform(query_img)

        # During training: randomly sample one positive satellite
        import random
        sat_path = random.choice(item["sat_paths"])
        sat_img = Image.open(sat_path).convert("RGB")
        if self.sat_transform:
            sat_img = self.sat_transform(sat_img)

        return {
            "query_img": query_img,
            "sat_img": sat_img,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "domain_id": torch.tensor(self.DOMAIN_ID, dtype=torch.long),
            "num_positives": torch.tensor(len(item["sat_paths"]), dtype=torch.long),
        }


class VIGORGallery(GalleryDataset):
    """All unique satellite tiles in VIGOR for gallery evaluation."""

    def __init__(self, root: str, split: str = "val", area: str = "same", transform=None):
        if transform is None:
            transform = val_transform()

        root = Path(root)
        json_path = root / "splits" / f"{area}-area" / f"{split}.json"
        with open(json_path) as f:
            data = json.load(f)

        # Collect all unique satellite tiles
        seen = {}
        for idx, (_, sat_ids) in enumerate(data.items()):
            for sid in sat_ids:
                sat_path = str(root / "satellite" / sid)
                if sat_path not in seen:
                    seen[sat_path] = len(seen)

        sat_paths = list(seen.keys())
        labels = list(seen.values())
        super().__init__(sat_paths, labels, transform)
