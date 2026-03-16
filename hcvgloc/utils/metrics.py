"""
metrics.py
-----------
Recall@K and mAP evaluation for HCVGLoc Stage 1 cross-view retrieval.

Recall@K:  fraction of queries where the correct satellite tile appears
           in the top-K retrieved results.

For VIGOR (multi-positive): a query is considered correct at K if ANY
of its positive satellites appears in the top-K.

Usage:
    scores = compute_recall_at_k(
        query_feats,   # [Nq, D] numpy or torch
        gallery_feats, # [Ng, D]
        query_labels,  # [Nq] int
        gallery_labels # [Ng] int
        k_vals=[1, 5, 10]
    )
    print(scores)  # {'R@1': 0.82, 'R@5': 0.94, 'R@10': 0.97}
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Union


def compute_recall_at_k(
    query_feats: Union[np.ndarray, torch.Tensor],
    gallery_feats: Union[np.ndarray, torch.Tensor],
    query_labels: Union[np.ndarray, torch.Tensor],
    gallery_labels: Union[np.ndarray, torch.Tensor],
    k_vals: List[int] = [1, 5, 10],
    batch_size: int = 256,
) -> dict:
    """
    Compute Recall@K for retrieval evaluation.

    Args:
        query_feats:    [Nq, D]  L2-normalised query descriptors
        gallery_feats:  [Ng, D]  L2-normalised gallery descriptors
        query_labels:   [Nq]     integer labels (GPS tile IDs)
        gallery_labels: [Ng]     integer labels for gallery
        k_vals:         list of K values to evaluate
        batch_size:     query batch size for memory-efficient computation

    Returns:
        dict  {'R@1': float, 'R@5': float, 'R@10': float, ...}
    """
    # Convert to torch tensors on CPU
    if isinstance(query_feats, np.ndarray):
        query_feats = torch.from_numpy(query_feats).float()
    if isinstance(gallery_feats, np.ndarray):
        gallery_feats = torch.from_numpy(gallery_feats).float()
    if isinstance(query_labels, np.ndarray):
        query_labels = torch.from_numpy(query_labels).long()
    if isinstance(gallery_labels, np.ndarray):
        gallery_labels = torch.from_numpy(gallery_labels).long()

    # L2 normalise (in case not already done)
    query_feats   = F.normalize(query_feats, p=2, dim=-1)
    gallery_feats = F.normalize(gallery_feats, p=2, dim=-1)

    Nq = query_feats.size(0)
    max_k = max(k_vals)
    correct = {k: 0 for k in k_vals}

    # Compute similarity in batches to avoid OOM on large galleries
    for start in range(0, Nq, batch_size):
        end = min(start + batch_size, Nq)
        q_batch = query_feats[start:end]          # [B, D]
        q_labels = query_labels[start:end]        # [B]

        # Cosine similarity [B, Ng]
        sim = torch.matmul(q_batch, gallery_feats.T)

        # Top-K retrieval
        topk_indices = torch.topk(sim, k=max_k, dim=-1).indices   # [B, max_k]
        topk_labels  = gallery_labels[topk_indices]                # [B, max_k]

        for k in k_vals:
            # Check if correct label appears in top-k for each query
            matches = (topk_labels[:, :k] == q_labels.unsqueeze(1))  # [B, k]
            correct[k] += matches.any(dim=1).sum().item()

    results = {f"R@{k}": correct[k] / Nq for k in k_vals}
    return results


@torch.no_grad()
def build_gallery_embeddings(
    model,
    gallery_loader,
    device: torch.device,
) -> tuple:
    """
    Build gallery embedding matrix from a GalleryDataset loader.

    Args:
        model:          CoarseLocalizer (or DDP-wrapped)
        gallery_loader: DataLoader over GalleryDataset
        device:         target device

    Returns:
        (embeddings [Ng, D], labels [Ng])  both as CPU tensors
    """
    # Unwrap DDP if needed
    m = model.module if hasattr(model, "module") else model
    m.eval()

    all_embs, all_labels = [], []
    for batch in gallery_loader:
        sat_imgs  = batch["sat_img"].to(device)
        labels    = batch["label"]

        embs = m.encode(sat_imgs)          # [B, D]
        all_embs.append(embs.cpu())
        all_labels.append(labels)

    return torch.cat(all_embs, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def build_query_embeddings(
    model,
    query_loader,
    device: torch.device,
) -> tuple:
    """
    Build query embedding matrix from a dataset loader.

    Returns:
        (embeddings [Nq, D], labels [Nq])  both as CPU tensors
    """
    m = model.module if hasattr(model, "module") else model
    m.eval()

    all_embs, all_labels = [], []
    for batch in query_loader:
        query_imgs = batch["query_img"].to(device)
        labels     = batch["label"]

        embs = m.encode(query_imgs)
        all_embs.append(embs.cpu())
        all_labels.append(labels)

    return torch.cat(all_embs, dim=0), torch.cat(all_labels, dim=0)
