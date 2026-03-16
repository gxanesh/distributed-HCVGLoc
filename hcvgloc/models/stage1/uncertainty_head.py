"""
uncertainty_head.py
--------------------
Aleatoric (data) uncertainty head for HCVGLoc Stage 1.

Predicts per-sample retrieval confidence (σ²) — a scalar variance that
indicates how certain the model is about its descriptor embedding.
Low σ² = high confidence; high σ² = low confidence.

Used downstream in:
  - Uncertainty-weighted InfoNCE loss during training
  - EKF observation noise R in Stage 3 temporal fusion

Based on heteroscedastic uncertainty estimation:
  Reference: Kendall & Gal, "What Uncertainties Do We Need in Bayesian
             Deep Learning for Computer Vision?", NeurIPS 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyHead(nn.Module):
    """
    Produces aleatoric uncertainty estimate (log σ²) from the descriptor.

    Predicts log variance for numerical stability; exponentiate to get σ².

    Args:
        descriptor_dim: input descriptor dimensionality (512)
        hidden_dim: MLP hidden width
    """

    def __init__(self, descriptor_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),   # scalar log-variance
        )
        # Clamp range to prevent σ² collapse or explosion
        self.log_var_min = -6.0
        self.log_var_max = 4.0

    def forward(self, descriptor: torch.Tensor):
        """
        Args:
            descriptor: [B, descriptor_dim]
        Returns:
            log_var: [B]   — log σ² (clamped)
            sigma2:  [B]   — σ² = exp(log_var)
        """
        log_var = self.head(descriptor).squeeze(-1)   # [B]
        log_var = torch.clamp(log_var, self.log_var_min, self.log_var_max)
        sigma2 = torch.exp(log_var)
        return log_var, sigma2

    def uncertainty_weighted_descriptor(
        self,
        descriptor: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weight descriptor by inverse uncertainty for downstream fusion.
        confidence = 1 / (σ² + ε)

        Args:
            descriptor: [B, D]
            sigma2: [B]
        Returns:
            weighted: [B, D]
        """
        confidence = 1.0 / (sigma2.unsqueeze(-1) + 1e-6)  # [B, 1]
        weighted = descriptor * confidence
        return F.normalize(weighted, p=2, dim=-1)
