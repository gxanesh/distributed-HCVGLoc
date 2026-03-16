"""
geometric_layout.py
--------------------
Geometric Layout Encoder (GLE) for HCVGLoc Stage 1.

Uses slot attention to produce a compact 512-D descriptor that captures
structural layout (roads, buildings, block geometry) rather than texture.
This makes the descriptor more robust to seasonal/lighting changes.

Architecture:
    Multi-scale BiFPN features {P3,P4,P5}
        → flatten & concat
        → Slot Attention (K=8 slots)
        → Pool slots → 512-D global descriptor
        → L2-normalise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SlotAttention(nn.Module):
    """
    Slot Attention module (Locatello et al., NeurIPS 2020).
    Learns K compact slot representations from N input tokens via
    iterative cross-attention.

    Args:
        num_slots: K (number of slots)
        slot_dim: dimensionality of each slot
        input_dim: dimensionality of input tokens
        num_iters: number of attention iterations
    """

    def __init__(
        self,
        num_slots: int = 8,
        slot_dim: int = 256,
        input_dim: int = 256,
        num_iters: int = 3,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iters = num_iters
        self.scale = slot_dim ** -0.5

        # Learnable slot initialisations
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(input_dim, slot_dim, bias=False)

        # GRU for iterative slot updates
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # Feed-forward after GRU
        self.ff = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, N, input_dim]  (N = number of spatial tokens)
        Returns:
            slots: [B, K, slot_dim]
        """
        B, N, _ = inputs.shape
        inputs = self.norm_input(inputs)

        # Initialise slots from learnable Gaussian
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)  # [B, K, slot_dim]

        k = self.to_k(inputs)   # [B, N, slot_dim]
        v = self.to_v(inputs)   # [B, N, slot_dim]

        for _ in range(self.num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.to_q(slots)  # [B, K, slot_dim]

            # Attention
            dots = torch.einsum("bkd,bnd->bkn", q, k) * self.scale  # [B, K, N]
            attn = dots.softmax(dim=1) + 1e-8                         # compete over slots
            attn = attn / attn.sum(dim=-1, keepdim=True)              # normalise

            # Aggregation
            updates = torch.einsum("bkn,bnd->bkd", attn, v)           # [B, K, slot_dim]

            # GRU update (slot_dim)
            slots = self.gru(
                updates.reshape(B * self.num_slots, self.slot_dim),
                slots_prev.reshape(B * self.num_slots, self.slot_dim),
            ).reshape(B, self.num_slots, self.slot_dim)

            slots = slots + self.ff(self.norm_pre_ff(slots))

        return slots   # [B, K, slot_dim]


class GeometricLayoutEncoder(nn.Module):
    """
    Full GLE module.

    1. Projects multi-scale BiFPN features to shared token dimension.
    2. Flattens & concatenates spatial tokens from {P3, P4, P5}.
    3. Runs slot attention to extract K structural slots.
    4. Pools slots → single 512-D descriptor.
    5. L2-normalises for cosine similarity retrieval.
    """

    def __init__(
        self,
        fpn_ch: int = 256,
        token_dim: int = 256,
        descriptor_dim: int = 512,
        num_slots: int = 8,
    ):
        super().__init__()

        # Project each FPN level to token_dim
        self.proj_p3 = nn.Conv2d(fpn_ch, token_dim, 1, bias=False)
        self.proj_p4 = nn.Conv2d(fpn_ch, token_dim, 1, bias=False)
        self.proj_p5 = nn.Conv2d(fpn_ch, token_dim, 1, bias=False)

        # Positional embedding (added to tokens)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, token_dim))

        self.slot_attn = SlotAttention(
            num_slots=num_slots,
            slot_dim=token_dim,
            input_dim=token_dim,
            num_iters=3,
        )

        # Project K pooled slots → descriptor_dim
        self.descriptor_head = nn.Sequential(
            nn.Linear(token_dim * num_slots, descriptor_dim * 2),
            nn.GELU(),
            nn.Linear(descriptor_dim * 2, descriptor_dim),
        )
        self.norm = nn.LayerNorm(descriptor_dim)

    def forward(self, fpn_features: dict) -> torch.Tensor:
        """
        Args:
            fpn_features: {'P3': [B,256,48,48], 'P4': [B,256,24,24], 'P5': [B,256,12,12]}
        Returns:
            descriptor: [B, descriptor_dim]  (L2-normalised)
        """
        # Project FPN levels → token_dim
        p3 = self.proj_p3(fpn_features["P3"])  # [B, token_dim, 48, 48]
        p4 = self.proj_p4(fpn_features["P4"])  # [B, token_dim, 24, 24]
        p5 = self.proj_p5(fpn_features["P5"])  # [B, token_dim, 12, 12]

        # Flatten each level to tokens
        def to_tokens(feat):
            B, C, H, W = feat.shape
            return rearrange(feat, "b c h w -> b (h w) c")

        t3 = to_tokens(p3)   # [B, 2304, token_dim]
        t4 = to_tokens(p4)   # [B,  576, token_dim]
        t5 = to_tokens(p5)   # [B,  144, token_dim]

        # Concatenate all spatial tokens
        tokens = torch.cat([t3, t4, t5], dim=1)   # [B, 3024, token_dim]
        tokens = tokens + self.pos_embed            # add learnable position bias

        # Slot attention → [B, K, token_dim]
        slots = self.slot_attn(tokens)

        # Pool slots: flatten K slots → single vector
        B, K, D = slots.shape
        pooled = slots.reshape(B, K * D)           # [B, K*token_dim]

        # Project to descriptor
        desc = self.descriptor_head(pooled)        # [B, descriptor_dim]
        desc = self.norm(desc)

        # L2 normalise
        desc = F.normalize(desc, p=2, dim=-1)
        return desc
