"""
university1652.py
------------------
University-1652 drone↔satellite cross-view loader for HCVGLoc Stage 1.

Dataset layout (Zheng et al. 2020):
    <root>/
        train/
            drone/<class_id>/image-XX.jpeg     # ~54 drone views per building
            satellite/<class_id>/<class_id>.jpg # 1 satellite image per building
            google/<class_id>/*.jpg             # unused here
            street/<class_id>/*.jpg             # unused here
        test/
            query_drone/<class_id>/image-XX.jpeg     # drone queries
            gallery_satellite/<class_id>/<class_id>.jpg  # satellite gallery (incl. distractors)
            query_satellite, gallery_drone, ...      # unused for drone→sat task

Train classes are disjoint from test classes (standard split).
Test gallery contains distractor classes not present in query_drone.

Label convention:
    class_id string "0839" → int(class_id) = 839
    Positive-pair mask in training: labels.unsqueeze(0) == labels.unsqueeze(1)
    R@K evaluation: query_label == gallery_label counts as correct.
"""

from pathlib import Path
from typing import Optional, Callable, List

from hcvgloc.datasets.base_dataset import CrossViewDataset, GalleryDataset
from hcvgloc.datasets.transforms import val_transform


_IMG_EXTS = (".jpg", ".jpeg", ".png")


def _list_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMG_EXTS)


class University1652Dataset(CrossViewDataset):
    """
    University-1652 drone→satellite cross-view dataset.

    Args:
        root:            path to University-1652 root directory
        split:           'train' or 'val'
                         'train' uses train/drone + train/satellite
                         'val'   uses test/query_drone + test/gallery_satellite
        query_transform: transform applied to drone (query) images
        sat_transform:   transform applied to satellite images
    """

    DOMAIN_ID = CrossViewDataset.DOMAIN_IDS["university1652"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        query_transform: Optional[Callable] = None,
        sat_transform: Optional[Callable] = None,
    ):
        super().__init__(root, split, query_transform, sat_transform)

    def _load_pairs(self):
        if self.split == "train":
            drone_root = self.root / "train" / "drone"
            sat_root   = self.root / "train" / "satellite"
        elif self.split == "val":
            drone_root = self.root / "test" / "query_drone"
            sat_root   = self.root / "test" / "gallery_satellite"
        else:
            raise ValueError(f"Unknown split '{self.split}'. Use 'train' or 'val'.")

        if not drone_root.exists():
            raise FileNotFoundError(f"University-1652 drone folder not found: {drone_root}")
        if not sat_root.exists():
            raise FileNotFoundError(f"University-1652 satellite folder not found: {sat_root}")

        for class_dir in sorted(drone_root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_id = class_dir.name
            sat_class_dir = sat_root / class_id
            if not sat_class_dir.exists():
                continue
            sat_images = _list_images(sat_class_dir)
            if not sat_images:
                continue
            sat_path = str(sat_images[0])
            label = int(class_id)
            for drone_img in _list_images(class_dir):
                self.pairs.append((str(drone_img), sat_path, label, self.DOMAIN_ID))


class University1652Gallery(GalleryDataset):
    """
    Satellite gallery for University-1652 R@K evaluation.

    Enumerates every class in test/gallery_satellite (including distractor classes
    that have no corresponding query). Label is int(class_id).
    """

    def __init__(self, root: str, split: str = "val", transform=None):
        if transform is None:
            transform = val_transform()

        root = Path(root)
        sat_root = {
            "train": root / "train" / "satellite",
            "val":   root / "test" / "gallery_satellite",
        }[split]

        if not sat_root.exists():
            raise FileNotFoundError(f"University-1652 satellite folder not found: {sat_root}")

        sat_paths: List[str] = []
        labels: List[int] = []
        for class_dir in sorted(sat_root.iterdir()):
            if not class_dir.is_dir():
                continue
            imgs = _list_images(class_dir)
            if not imgs:
                continue
            sat_paths.append(str(imgs[0]))
            labels.append(int(class_dir.name))

        super().__init__(sat_paths, labels, transform)
