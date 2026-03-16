"""
coarse_localizer.py
--------------------
Full HCVGLoc Stage 1 model: Coarse Retrieval.

Wires together:
    EfficientViT-B1 Backbone
        → BiFPN Neck
            → Geometric Layout Encoder (GLE)  → 512-D descriptor
            → GRL + Domain Classifier          → domain logits
            → Rotation Module                  → (sinθ, cosθ), rot-aware desc
            → Uncertainty Head                 → σ²

The model is designed for DistributedDataParallel (DDP) training.
All-reduce is handled externally by DDP; no special synchronisation
primitives are needed inside this module.

Usage:
    model = CoarseLocalizer()

    # Single forward pass (returns HCVGLocOutput dataclass)
    out = model(query_img, satellite_img, domain_label)

    # Descriptor-only (inference / gallery build)
    desc = model.encode(img)
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from hcvgloc.models.backbone.efficientvit import EfficientViTEncoder
from hcvgloc.models.stage1.bifpn import BiFPN
from hcvgloc.models.stage1.geometric_layout import GeometricLayoutEncoder
from hcvgloc.models.stage1.domain_adversarial import DomainAdversarialModule
from hcvgloc.models.stage1.rotation_module import RotationModule
from hcvgloc.models.stage1.uncertainty_head import UncertaintyHead


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class Stage1Output:
    """Named outputs from a single CoarseLocalizer forward pass."""

    # Descriptors
    query_desc: torch.Tensor        # [B, 512]  L2-normalised
    sat_desc: torch.Tensor          # [B, 512]  L2-normalised

    # Auxiliary heads — query side
    domain_logits: torch.Tensor     # [B, num_domains]
    sin_cos: torch.Tensor           # [B, 2]
    log_var: torch.Tensor           # [B]
    sigma2: torch.Tensor            # [B]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CoarseLocalizer(nn.Module):
    """
    HCVGLoc Stage 1: Coarse Retrieval Model.

    Shared-weight encoder: the same EfficientViT-B1 + BiFPN + GLE processes
    both query (UAV) and satellite images. Only the GRL/domain classifier,
    rotation head, and uncertainty head are query-only branches.

    Args:
        descriptor_dim:  output descriptor dimensionality (default 512)
        fpn_out_ch:      BiFPN unified channel width (default 256)
        num_fpn_layers:  number of BiFPN stacks (default 3)
        num_slots:       slot attention slots in GLE (default 8)
        num_domains:     dataset domains for GRL classifier (default 4)
        grl_lambda:      initial GRL lambda (default 0.0, ramped by trainer)
    """

    def __init__(
        self,
        descriptor_dim: int = 512,
        fpn_out_ch: int = 256,
        num_fpn_layers: int = 3,
        num_slots: int = 8,
        num_domains: int = 4,
        grl_lambda: float = 0.0,
    ):
        super().__init__()
        self.descriptor_dim = descriptor_dim

        # --- Shared backbone & neck ---
        self.backbone = EfficientViTEncoder()
        self.neck = BiFPN(
            in_channels={"C3": 128, "C4": 256, "C5": 512},
            out_ch=fpn_out_ch,
            num_layers=num_fpn_layers,
        )

        # --- Descriptor head (shared query & satellite) ---
        self.gle = GeometricLayoutEncoder(
            fpn_ch=fpn_out_ch,
            token_dim=256,
            descriptor_dim=descriptor_dim,
            num_slots=num_slots,
        )

        # --- Query-specific heads ---
        self.domain_head = DomainAdversarialModule(
            descriptor_dim=descriptor_dim,
            num_domains=num_domains,
            lambda_init=grl_lambda,
        )
        self.rotation_head = RotationModule(
            descriptor_dim=descriptor_dim,
        )
        self.uncertainty_head = UncertaintyHead(
            descriptor_dim=descriptor_dim,
        )

    # -----------------------------------------------------------------------
    # Shared encoder: backbone + neck + GLE
    # -----------------------------------------------------------------------

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode a single image batch to a 512-D L2-normalised descriptor.

        Args:
            img: [B, 3, 384, 384]
        Returns:
            descriptor: [B, 512]
        """
        feats = self.backbone(img)    # {'C3','C4','C5'}
        fpn = self.neck(feats)        # {'P3','P4','P5'}
        desc = self.gle(fpn)          # [B, 512]
        return desc

    # -----------------------------------------------------------------------
    # Full forward (training)
    # -----------------------------------------------------------------------

    def forward(
        self,
        query_img: torch.Tensor,
        sat_img: torch.Tensor,
        domain_label: Optional[torch.Tensor] = None,
    ) -> Stage1Output:
        """
        Args:
            query_img:    [B, 3, 384, 384]
            sat_img:      [B, 3, 384, 384]
            domain_label: [B] int64 domain IDs (0-3), used only in loss
        Returns:
            Stage1Output with all head outputs
        """
        # Encode both views
        q_desc = self.encode(query_img)    # [B, 512]
        s_desc = self.encode(sat_img)      # [B, 512]

        # Query auxiliary heads
        domain_logits = self.domain_head(q_desc)         # [B, num_domains]
        rot_desc, sin_cos = self.rotation_head(q_desc)   # [B,512], [B,2]
        log_var, sigma2 = self.uncertainty_head(q_desc)  # [B], [B]

        # Use rotation-aware descriptor as final query descriptor
        return Stage1Output(
            query_desc=rot_desc,
            sat_desc=s_desc,
            domain_logits=domain_logits,
            sin_cos=sin_cos,
            log_var=log_var,
            sigma2=sigma2,
        )

    # -----------------------------------------------------------------------
    # GRL lambda scheduling (called by trainer each epoch)
    # -----------------------------------------------------------------------

    def set_grl_lambda(self, lambda_: float):
        self.domain_head.set_lambda(lambda_)

    def get_grl_lambda(self) -> float:
        return self.domain_head.grl.lambda_

    # -----------------------------------------------------------------------
    # Parameter groups for differential learning rates
    # -----------------------------------------------------------------------

    def get_param_groups(self, base_lr: float = 3e-4, backbone_lr_scale: float = 0.1):
        """
        Returns parameter groups with lower LR for pretrained backbone.
        """
        backbone_params = list(self.backbone.parameters())
        backbone_ids = set(id(p) for p in backbone_params)
        other_params = [p for p in self.parameters() if id(p) not in backbone_ids]

        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_scale},
            {"params": other_params, "lr": base_lr},
        ]


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoarseLocalizer().to(device)

    B = 4
    q = torch.randn(B, 3, 384, 384, device=device)
    s = torch.randn(B, 3, 384, 384, device=device)

    out = model(q, s)
    print(f"query_desc:    {out.query_desc.shape}")    # [4, 512]
    print(f"sat_desc:      {out.sat_desc.shape}")      # [4, 512]
    print(f"domain_logits: {out.domain_logits.shape}") # [4, 4]
    print(f"sin_cos:       {out.sin_cos.shape}")       # [4, 2]
    print(f"log_var:       {out.log_var.shape}")       # [4]

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nTotal params: {total:.1f}M")
