"""
infonce.py
-----------
Label-aware InfoNCE loss for HCVGLoc Stage 1 coarse retrieval.

Standard InfoNCE assumes exactly one positive per anchor. VIGOR and
University-1652 have multiple ground-truth positives per query. This
implementation supports multi-positive contrastive learning by masking
all positives from the denominator (treating them all as signal).

Formula:
    L = -1/B · Σᵢ log [
            Σⱼ∈Pᵢ exp(sᵢⱼ / τ)
        ─────────────────────────────────────────────────────
        Σⱼ∈Pᵢ exp(sᵢⱼ / τ) + Σₖ∉Pᵢ exp(sᵢₖ / τ)
    ]

where:
    sᵢⱼ = cosine similarity between query i and satellite j
    Pᵢ  = set of positive (matching) satellites for query i
    τ   = temperature (default 0.07 — see training gap analysis)

Reference:
    A Simple Framework for Contrastive Learning (SimCLR, Chen et al. 2020)
    Multi-positive extension: Khosla et al., Supervised Contrastive Learning (NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelAwareInfoNCE(nn.Module):
    """
    Multi-positive InfoNCE loss.

    Args:
        temperature:   τ (default 0.07 — fixes the τ=0.02 overfitting issue)
        max_violation: if True, use only the hardest negative per query
                       (equivalent to triplet loss with hard mining)
    """

    def __init__(self, temperature: float = 0.07, max_violation: bool = False):
        super().__init__()
        self.tau = temperature
        self.max_violation = max_violation

    def forward(
        self,
        query_desc: torch.Tensor,
        sat_desc: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_desc: [B, D]  L2-normalised query descriptors
            sat_desc:   [B, D]  L2-normalised satellite descriptors
                                (rows correspond to gallery entries)
            pos_mask:   [B, B]  bool — pos_mask[i,j]=True if sat j is a
                                positive match for query i

        Returns:
            loss: scalar InfoNCE loss
        """
        # Cosine similarity matrix [B, B]
        sim = torch.matmul(query_desc, sat_desc.T) / self.tau

        # Numerical stability: subtract row max (log-sum-exp trick)
        sim = sim - sim.max(dim=-1, keepdim=True).values.detach()

        exp_sim = torch.exp(sim)                   # [B, B]

        # Denominator: sum over all pairs (positives + negatives)
        denom = exp_sim.sum(dim=-1, keepdim=True)  # [B, 1]

        # Numerator: sum over positives only
        pos_mask = pos_mask.float()
        if self.max_violation:
            # Hard mining: use only hardest negative per query
            # (zero out positives, take max negative)
            neg_sim = sim.masked_fill(pos_mask.bool(), -1e9)
            hardest_neg = neg_sim.max(dim=-1, keepdim=True).values
            # Use single hardest positive
            pos_sim = sim.masked_fill(~pos_mask.bool(), -1e9)
            hardest_pos = pos_sim.max(dim=-1, keepdim=True).values
            loss = F.softplus(hardest_neg - hardest_pos).mean()
            return loss

        # Standard multi-positive InfoNCE
        # log P(positive | query) = log [Σ_pos exp(s) / Σ_all exp(s)]
        pos_count = pos_mask.sum(dim=-1).clamp(min=1)   # [B]
        log_pos = (exp_sim * pos_mask).sum(dim=-1)       # [B]
        log_pos = torch.log(log_pos + 1e-8)
        log_denom = torch.log(denom.squeeze(-1) + 1e-8)

        loss = -(log_pos - log_denom) / pos_count
        return loss.mean()

    def build_pos_mask(
        self,
        labels: torch.Tensor,
        tol: float = 0.0,
    ) -> torch.Tensor:
        """
        Build positive mask from integer class labels.

        Args:
            labels: [B] int64 class labels (or GPS tile IDs)
            tol:    not used; kept for API consistency

        Returns:
            pos_mask: [B, B] bool
        """
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        return pos_mask
