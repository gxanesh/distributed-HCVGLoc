"""
composite_loss.py
------------------
Combined Stage 1 training loss for HCVGLoc.

    L_total = λ_nce · L_InfoNCE
            + λ_js  · L_JS
            + λ_dom · L_domain
            + λ_rot · L_rotation
            + λ_unc · L_uncertainty

Default weights (from ablation study):
    λ_nce = 1.0  (primary retrieval signal)
    λ_js  = 0.5  (distribution alignment)
    λ_dom = 0.1  (domain adversarial — only after GRL warmup)
    λ_rot = 0.2  (rotation consistency)
    λ_unc = 0.1  (uncertainty calibration)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hcvgloc.losses.infonce import LabelAwareInfoNCE, MemoryBankInfoNCE


class JSDiv(nn.Module):
    """Jensen-Shannon divergence between query and satellite similarity distributions."""

    def forward(
        self,
        query_desc: torch.Tensor,
        sat_desc: torch.Tensor,
    ) -> torch.Tensor:
        q_sim = torch.matmul(query_desc, query_desc.T)    # [B, B]
        s_sim = torch.matmul(sat_desc, sat_desc.T)        # [B, B]

        p = F.softmax(q_sim, dim=-1)
        q = F.softmax(s_sim, dim=-1)
        m = 0.5 * (p + q)

        js = 0.5 * (
            F.kl_div(m.log(), p, reduction="batchmean") +
            F.kl_div(m.log(), q, reduction="batchmean")
        )
        return js


class RotationLoss(nn.Module):
    """
    Rotation consistency loss.

    L_rot = 1 - cos_sim(descriptor(I), descriptor(R_θ(I)))

    In practice, we supervise the (sin θ, cos θ) prediction against the
    ground-truth relative rotation angle when annotations are available,
    and use consistency loss (self-supervised) otherwise.
    """

    def forward(
        self,
        sin_cos_pred: torch.Tensor,
        sin_cos_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sin_cos_pred: [B, 2]  predicted (sin θ, cos θ)
            sin_cos_gt:   [B, 2]  ground-truth (sin θ, cos θ) or zeros if unknown
        Returns:
            scalar loss
        """
        # Cosine similarity on unit circle → 1 when perfectly aligned
        cos_sim = F.cosine_similarity(sin_cos_pred, sin_cos_gt, dim=-1)
        return (1.0 - cos_sim).mean()


class UncertaintyLoss(nn.Module):
    """
    Negative log-likelihood uncertainty calibration loss.

    L_unc = 0.5 · (||d_q - d_p||² / σ² + log σ²)

    Forces σ² to be large when descriptors are far apart (hard samples)
    and small when they are close (easy samples).
    """

    def forward(
        self,
        query_desc: torch.Tensor,
        pos_desc: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_desc: [B, D]
            pos_desc:   [B, D]  positive (matching) satellite descriptor
            log_var:    [B]     predicted log σ²
        Returns:
            scalar loss
        """
        sq_dist = (query_desc - pos_desc).pow(2).sum(dim=-1)   # [B]
        variance = torch.exp(log_var)                          # [B]
        nll = 0.5 * (sq_dist / (variance + 1e-8) + log_var)
        return nll.mean()


class Stage1Loss(nn.Module):
    """
    Full composite loss for HCVGLoc Stage 1 training.

    Args:
        temperature:   InfoNCE temperature τ (default 0.07)
        lambda_nce:    weight for InfoNCE
        lambda_js:     weight for JS divergence
        lambda_domain: weight for domain adversarial loss
        lambda_rot:    weight for rotation loss
        lambda_unc:    weight for uncertainty loss
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_nce: float = 1.0,
        lambda_js: float = 0.5,
        lambda_domain: float = 0.1,
        lambda_rot: float = 0.2,
        lambda_unc: float = 0.1,
        # Optional memory-bank / hard-negative-aware InfoNCE settings.
        # Activated when use_memory_bank=True or hard_negative_weight>0.
        use_memory_bank: bool = False,
        memory_bank_size: int = 8192,
        feature_dim: int = 512,
        infonce_symmetric: bool = True,
        infonce_hard_negative_weight: float = 0.0,
        infonce_hard_negative_margin: float = 0.3,
        infonce_label_smoothing: float = 0.0,
        # Label smoothing for the domain-classifier cross-entropy.
        domain_label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_nce = lambda_nce
        self.lambda_js = lambda_js
        self.lambda_domain = lambda_domain
        self.lambda_rot = lambda_rot
        self.lambda_unc = lambda_unc

        if use_memory_bank or infonce_hard_negative_weight > 0:
            self.nce_loss = MemoryBankInfoNCE(
                temperature=temperature,
                symmetric=infonce_symmetric,
                hard_negative_weight=infonce_hard_negative_weight,
                hard_negative_margin=infonce_hard_negative_margin,
                use_memory_bank=use_memory_bank,
                feature_dim=feature_dim,
                bank_size=memory_bank_size,
                label_smoothing=infonce_label_smoothing,
            )
        else:
            self.nce_loss = LabelAwareInfoNCE(temperature=temperature)
        self.js_loss = JSDiv()
        self.domain_loss = nn.CrossEntropyLoss(
            label_smoothing=domain_label_smoothing
        )
        self.rot_loss = RotationLoss()
        self.unc_loss = UncertaintyLoss()

    def forward(
        self,
        query_desc: torch.Tensor,      # [B, D]
        sat_desc: torch.Tensor,        # [B, D]
        pos_mask: torch.Tensor,        # [B, B] bool
        domain_logits: torch.Tensor,   # [B, num_domains]
        domain_labels: torch.Tensor,   # [B] int64
        sin_cos_pred: torch.Tensor,    # [B, 2]
        log_var: torch.Tensor,         # [B]
        sin_cos_gt: torch.Tensor = None,   # [B, 2] or None
    ) -> dict:
        """
        Computes all loss components and returns a dict for logging.

        Returns:
            {
                'total': scalar total loss,
                'nce': InfoNCE component,
                'js': JS divergence component,
                'domain': domain adversarial component,
                'rotation': rotation component,
                'uncertainty': uncertainty component,
            }
        """
        # 1. InfoNCE (primary retrieval loss)
        l_nce = self.nce_loss(query_desc, sat_desc, pos_mask)

        # 2. JS divergence (distribution alignment)
        l_js = self.js_loss(query_desc, sat_desc)

        # 3. Domain adversarial
        l_domain = self.domain_loss(domain_logits, domain_labels)

        # 4. Rotation consistency
        if sin_cos_gt is None:
            # No GT orientation → skip rotation loss
            l_rot = torch.tensor(0.0, device=query_desc.device)
        else:
            l_rot = self.rot_loss(sin_cos_pred, sin_cos_gt)

        # 5. Uncertainty calibration
        # For uncertainty loss, use the positive satellite per query
        # (take diagonal of similarity matrix as positive pairs in a batch)
        pos_idx = pos_mask.float().argmax(dim=-1)       # [B] — first positive
        pos_desc = sat_desc[pos_idx]                    # [B, D]
        l_unc = self.unc_loss(query_desc, pos_desc, log_var)

        # Weighted sum
        total = (
            self.lambda_nce * l_nce
            + self.lambda_js * l_js
            + self.lambda_domain * l_domain
            + self.lambda_rot * l_rot
            + self.lambda_unc * l_unc
        )

        return {
            "total": total,
            "nce": l_nce.detach(),
            "js": l_js.detach(),
            "domain": l_domain.detach(),
            "rotation": l_rot.detach() if torch.is_tensor(l_rot) else l_rot,
            "uncertainty": l_unc.detach(),
        }
