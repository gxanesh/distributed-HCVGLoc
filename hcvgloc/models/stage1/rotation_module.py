"""
rotation_module.py
-------------------
Rotation-aware head for HCVGLoc Stage 1.

The satellite and ground-facing UAV images may be captured at different
orientations. This head regresses the relative rotation (θ) as (sin θ, cos θ)
— a smooth, discontinuity-free parameterisation — and enforces rotational
consistency via a descriptor-level loss.

Output:
    sin_cos: [B, 2]  — predicted (sin θ, cos θ) for orientation estimation
    rot_descriptor: [B, descriptor_dim]  — rotation-augmented descriptor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotationModule(nn.Module):
    """
    Rotation-aware descriptor head.

    Takes the GLE descriptor and produces:
      1. A rotation-aware refined descriptor (adds orientation context).
      2. A (sin θ, cos θ) rotation prediction for explicit supervision.
    """

    def __init__(self, descriptor_dim: int = 512, hidden_dim: int = 256):
        super().__init__()

        # Rotation regression branch
        self.rot_head = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),   # outputs (sin θ, cos θ)
        )

        # Orientation embedding: maps (sin θ, cos θ) back to descriptor space
        self.orient_embed = nn.Linear(2, descriptor_dim)

        # Refinement MLP: fuses original descriptor with orientation embedding
        self.refine = nn.Sequential(
            nn.Linear(descriptor_dim * 2, descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, descriptor_dim),
        )
        self.norm = nn.LayerNorm(descriptor_dim)

    def forward(self, descriptor: torch.Tensor):
        """
        Args:
            descriptor: [B, descriptor_dim]  (L2-normalised from GLE)
        Returns:
            rot_descriptor: [B, descriptor_dim]  L2-normalised, rotation-aware
            sin_cos: [B, 2]  predicted rotation (sin θ, cos θ)
        """
        # Predict rotation
        sin_cos_raw = self.rot_head(descriptor)   # [B, 2]
        # Normalise to lie on unit circle: ensures ‖(sinθ, cosθ)‖ = 1
        sin_cos = F.normalize(sin_cos_raw, p=2, dim=-1)   # [B, 2]

        # Map orientation back to descriptor space
        orient_embed = self.orient_embed(sin_cos)  # [B, descriptor_dim]

        # Fuse with original descriptor
        fused = torch.cat([descriptor, orient_embed], dim=-1)  # [B, 2*D]
        rot_desc = self.refine(fused)              # [B, descriptor_dim]
        rot_desc = self.norm(rot_desc)

        # L2 normalise
        rot_desc = F.normalize(rot_desc, p=2, dim=-1)

        return rot_desc, sin_cos
