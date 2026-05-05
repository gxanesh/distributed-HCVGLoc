"""
base_dataset.py
----------------
Abstract base class for all cross-view geo-localization datasets.

Every dataset (CVUSA, CVACT, VIGOR) inherits from CrossViewDataset and
implements:
    - _load_pairs():  populate self.pairs list of (query_path, sat_path, label, domain_id)
    - __len__()
    - __getitem__()

The base class handles:
    - Image loading and transforms
    - DDP-aware DistributedSampler compatibility
    - Domain label assignment
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class CrossViewDataset(Dataset, ABC):
    """
    Abstract base for cross-view geo-localization datasets.

    Each sample returns:
        query_img:   [3, H, W] tensor (UAV / ground-facing view)
        sat_img:     [3, H, W] tensor (satellite / aerial view)
        label:       int — GPS tile ID or class index
        domain_id:   int — dataset domain (0=CVUSA,1=VIGOR,2=CVACT,3=Univ1652)
    """

    DOMAIN_IDS = {
        "cvusa": 0,
        "vigor": 1,
        "cvact": 2,
        "university1652": 3,
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        query_transform: Optional[Callable] = None,
        sat_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.query_transform = query_transform
        self.sat_transform = sat_transform

        self.pairs: List[Tuple] = []   # filled by _load_pairs()
        self._load_pairs()

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] No pairs found in {root} for split={split}. "
                "Check dataset path and split name."
            )

    @abstractmethod
    def _load_pairs(self):
        """Populate self.pairs with (query_path, sat_path, label, domain_id)."""
        pass

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        query_path, sat_path, label, domain_id = self.pairs[idx]

        query_img = self._load_image(query_path)
        sat_img = self._load_image(sat_path)

        if self.query_transform:
            query_img = self.query_transform(query_img)
        if self.sat_transform:
            sat_img = self.sat_transform(sat_img)

        return {
            "query_img": query_img,
            "sat_img": sat_img,
            "label": torch.tensor(label, dtype=torch.long),
            "domain_id": torch.tensor(domain_id, dtype=torch.long),
        }


class GalleryDataset(Dataset):
    """
    Satellite-only gallery dataset for Recall@K evaluation.

    Loads all unique satellite images without pairing.
    Used to build the full gallery embedding matrix at validation time.
    """

    def __init__(self, sat_paths: List[str], labels: List[int], transform=None):
        self.sat_paths = sat_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.sat_paths)

    def __getitem__(self, idx):
        img = Image.open(self.sat_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {
            "sat_img": img,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
        }
