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


# ─────────────────────────────────────────────────────────────────────────────
# Memory bank + hard-negative-aware InfoNCE
# Ported from gxanesh/avgl_plus_plus avgl/losses/__init__.py (commit f261fae).
# Adapted to dist-HCVGLoc's pos_mask API and extended so the bank also
# contributes negatives in the label-aware (multi-positive) path — the upstream
# code dropped the bank in that branch, so the bank was effectively a no-op
# for class-based datasets like University-1652.
# ─────────────────────────────────────────────────────────────────────────────

class MemoryBank(nn.Module):
    """
    FIFO buffer of past reference (satellite) descriptors. Increases the
    effective negative pool from `batch_size` to `bank_size` at the cost of
    a single normalised tensor on the device.
    """

    def __init__(self, feature_dim: int = 512, bank_size: int = 8192):
        super().__init__()
        self.bank_size = bank_size
        self.feature_dim = feature_dim
        self.register_buffer(
            "bank",
            F.normalize(torch.randn(bank_size, feature_dim), dim=1),
        )
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update(self, features: torch.Tensor):
        features = F.normalize(features.detach(), dim=1)
        b = features.shape[0]
        ptr = int(self.ptr)
        if ptr + b > self.bank_size:
            tail = self.bank_size - ptr
            self.bank[ptr:] = features[:tail]
            self.bank[: b - tail] = features[tail:]
        else:
            self.bank[ptr : ptr + b] = features
        self.ptr[0] = (ptr + b) % self.bank_size

    def get(self, device: torch.device = None) -> torch.Tensor:
        # Return a detached clone so the in-place `update()` later in the
        # same step doesn't mutate storage that autograd still needs for
        # backward. Cheap on a 8192×512 buffer (≈16 MB).
        snap = self.bank if (device is None or self.bank.device == device) else self.bank.to(device)
        return snap.detach().clone()


class MemoryBankInfoNCE(nn.Module):
    """
    Multi-positive InfoNCE with optional memory bank and hard-negative mining.

    Args:
        temperature:           softmax temperature τ
        symmetric:             also compute the r→q direction (in-batch only;
                               bank is asymmetric by construction)
        hard_negative_weight:  weight on the triplet-margin term
                               (set to 0 to disable)
        hard_negative_margin:  margin in `relu(hardest_neg - hardest_pos + m)`
        use_memory_bank:       enable cross-batch negatives
        feature_dim:           descriptor dim (only used if use_memory_bank)
        bank_size:             number of past references to keep
        label_smoothing:       smoothing strength applied to the soft target
                               distribution (0 disables)

    Notes:
        * Label-aware throughout — `pos_mask` may flag any number of
          positives per query, including > 1.
        * The bank receives `sat_desc.detach()` after the loss is computed,
          so its entries are gradient-free past references.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        symmetric: bool = True,
        hard_negative_weight: float = 0.5,
        hard_negative_margin: float = 0.3,
        use_memory_bank: bool = False,
        feature_dim: int = 512,
        bank_size: int = 8192,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.tau = temperature
        self.symmetric = symmetric
        self.hard_negative_weight = hard_negative_weight
        self.hard_negative_margin = hard_negative_margin
        self.use_memory_bank = use_memory_bank
        self.label_smoothing = label_smoothing
        self.ref_bank = (
            MemoryBank(feature_dim, bank_size) if use_memory_bank else None
        )

    def forward(
        self,
        query_desc: torch.Tensor,
        sat_desc: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_desc = F.normalize(query_desc, dim=-1)
        sat_desc = F.normalize(sat_desc, dim=-1)
        device = query_desc.device
        pos_mask = pos_mask.bool()
        B = query_desc.size(0)

        sim = torch.matmul(query_desc, sat_desc.T)  # [B, B]

        bank_sim = None
        if self.use_memory_bank and self.ref_bank is not None:
            bank_feats = self.ref_bank.get(device=device)        # [N, D]
            bank_sim = torch.matmul(query_desc, bank_feats.T)    # [B, N]

        # ── q→r direction (with bank as extra negatives) ─────────────────
        if bank_sim is not None:
            combined = torch.cat([sim, bank_sim], dim=1)         # [B, B+N]
            ext_pos = torch.cat(
                [pos_mask, torch.zeros(B, bank_sim.size(1), dtype=torch.bool, device=device)],
                dim=1,
            )
        else:
            combined = sim
            ext_pos = pos_mask

        loss = self._multi_positive_xent(combined / self.tau, ext_pos)

        # ── r→q direction (in-batch only — symmetric path) ───────────────
        if self.symmetric:
            loss_rq = self._multi_positive_xent(sim.T / self.tau, pos_mask.T)
            loss = 0.5 * (loss + loss_rq)

        # ── hard-negative margin term ────────────────────────────────────
        if self.hard_negative_weight > 0:
            loss = loss + self.hard_negative_weight * self._hard_neg_margin(
                sim, bank_sim, pos_mask
            )

        # Update bank AFTER computing loss so this step's positives are not
        # also treated as negatives via the bank.
        if self.use_memory_bank and self.ref_bank is not None:
            self.ref_bank.update(sat_desc)

        return loss

    def _multi_positive_xent(
        self,
        logits: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-positive cross-entropy. Numerator averages over positives,
        denominator sums over all targets. Optional label smoothing blends
        the positive log-prob with a uniform-target log-prob.
        """
        logits = logits - logits.max(dim=-1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits)
        log_denom = torch.log(exp_logits.sum(dim=-1) + 1e-8)         # [B]

        pos_count = pos_mask.float().sum(dim=-1).clamp(min=1)        # [B]
        # Mean log-prob over positives (multi-positive: average within row)
        pos_log = (logits * pos_mask.float()).sum(dim=-1) / pos_count
        loss_pos = -(pos_log - log_denom)

        if self.label_smoothing > 0:
            # Uniform target component: mean logit minus log_denom
            uni_log = logits.mean(dim=-1)
            loss_uni = -(uni_log - log_denom)
            return ((1 - self.label_smoothing) * loss_pos
                    + self.label_smoothing * loss_uni).mean()
        return loss_pos.mean()

    def _hard_neg_margin(
        self,
        sim: torch.Tensor,
        bank_sim: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """`relu(max_neg_sim - max_pos_sim + margin)` with bank in the neg pool."""
        # Hardest positive among in-batch positives
        pos_filled = sim.masked_fill(~pos_mask, -65000.0)
        hardest_pos = pos_filled.max(dim=-1).values                  # [B]

        # Hardest negative among in-batch negatives + bank
        neg_filled = sim.masked_fill(pos_mask, -65000.0)
        if bank_sim is not None:
            neg_filled = torch.cat([neg_filled, bank_sim], dim=1)
        hardest_neg = neg_filled.max(dim=-1).values                  # [B]

        return F.relu(hardest_neg - hardest_pos + self.hard_negative_margin).mean()
