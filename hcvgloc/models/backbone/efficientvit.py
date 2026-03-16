"""
efficientvit.py
---------------
Customized EfficientViT-B1 backbone for HCVGLoc Stage 1.

Architecture:
    Stem (384→96) → Stage1 (96→48, 64-D) → Stage2 (48→24, 128-D)
    → Stage3 (24→12, 256-D) → Stage4 (12→12, 512-D)

Outputs multi-scale feature maps {C3, C4, C5} for the BiFPN neck.
Uses linear O(N) attention in place of standard O(N²) self-attention,
giving ~6× speedup over ViT-Base with only ~1-2% accuracy loss.

Reference: EfficientViT: Memory Efficient Vision Transformer with
           Cascaded Group Attention (Liu et al., CVPR 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Linear Attention (O(N) complexity)
# ---------------------------------------------------------------------------

class LinearAttention(nn.Module):
    """
    Linear attention using ReLU feature map: φ(x) = ReLU(x).
    Complexity: O(Nd²) instead of O(N²d).

    Decomposition:  Attention(Q,K,V) ≈ φ(Q) · (φ(K)ᵀ · V)
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]  (N = sequence length = H*W after patching)
        Returns:
            [B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)           # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)                     # each [B, heads, N, head_dim]

        # Feature map: φ(x) = ReLU(x) + ε
        q = F.relu(q) + 1e-6
        k = F.relu(k) + 1e-6

        # Linear attention:  O(Nd²)
        # kv = φ(K)ᵀ · V  →  [B, heads, head_dim, head_dim]
        kv = torch.einsum("bhnd,bhnc->bhdc", k, v)
        # out = φ(Q) · kv  →  [B, heads, N, head_dim]
        out = torch.einsum("bhnd,bhdc->bhnc", q, kv)

        # Normalise
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6)
        out = out * z.unsqueeze(-1)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.proj(out)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        layers = [
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.GELU())
        super().__init__(*layers)


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (used in EfficientViT stages)."""

    def __init__(self, in_ch: int, out_ch: int, expand: int = 4, stride: int = 1):
        super().__init__()
        mid_ch = in_ch * expand
        self.conv = nn.Sequential(
            ConvBNAct(in_ch, mid_ch, k=1, p=0),
            ConvBNAct(mid_ch, mid_ch, k=3, s=stride, p=1),   # depthwise-like
            ConvBNAct(mid_ch, out_ch, k=1, p=0, act=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class EfficientViTBlock(nn.Module):
    """
    One EfficientViT block:
        MBConv → flatten → LinearAttention → reshape → MBConv (FFN)
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.mbconv1 = MBConv(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, num_heads)
        self.mbconv2 = MBConv(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W]"""
        B, C, H, W = x.shape
        x = self.mbconv1(x)

        # Spatial → sequence
        tokens = rearrange(x, "b c h w -> b (h w) c")
        tokens = self.norm(tokens)
        tokens = tokens + self.attn(tokens)
        x = rearrange(tokens, "b (h w) c -> b c h w", h=H, w=W)

        return self.mbconv2(x)


# ---------------------------------------------------------------------------
# Full EfficientViT-B1 Encoder
# ---------------------------------------------------------------------------

class EfficientViTEncoder(nn.Module):
    """
    Customised EfficientViT-B1 encoder for HCVGLoc.

    Stage output channels (B1 config):
        Stem  : 64-D  @ H/4  (96×96 for 384 input)
        Stage1: 128-D @ H/8  (48×48)  → C3
        Stage2: 256-D @ H/16 (24×24)  → C4
        Stage3: 512-D @ H/32 (12×12)  → C5

    Returns:
        dict with keys 'C3', 'C4', 'C5'
    """

    CHANNELS = {"stem": 64, "s1": 128, "s2": 256, "s3": 512}

    def __init__(self, pretrained: bool = False):
        super().__init__()
        ch = self.CHANNELS

        # Stem: 3 → 64,  384→96 (stride 4)
        self.stem = nn.Sequential(
            ConvBNAct(3, ch["stem"] // 2, k=3, s=2, p=1),   # 384→192
            ConvBNAct(ch["stem"] // 2, ch["stem"], k=3, s=2, p=1),  # 192→96
        )

        # Stage 1: 64→128, stride 2  (96→48)
        self.stage1 = nn.Sequential(
            MBConv(ch["stem"], ch["s1"], stride=2),
            EfficientViTBlock(ch["s1"], num_heads=4),
            EfficientViTBlock(ch["s1"], num_heads=4),
        )

        # Stage 2: 128→256, stride 2  (48→24)
        self.stage2 = nn.Sequential(
            MBConv(ch["s1"], ch["s2"], stride=2),
            EfficientViTBlock(ch["s2"], num_heads=8),
            EfficientViTBlock(ch["s2"], num_heads=8),
            EfficientViTBlock(ch["s2"], num_heads=8),
        )

        # Stage 3: 256→512, stride 1  (24→12 via pool inside MBConv)
        self.stage3 = nn.Sequential(
            MBConv(ch["s2"], ch["s3"], stride=2),
            EfficientViTBlock(ch["s3"], num_heads=16),
            EfficientViTBlock(ch["s3"], num_heads=16),
            EfficientViTBlock(ch["s3"], num_heads=16),
            EfficientViTBlock(ch["s3"], num_heads=16),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, 3, 384, 384]
        Returns:
            {'C3': [B,128,48,48], 'C4': [B,256,24,24], 'C5': [B,512,12,12]}
        """
        x = self.stem(x)       # [B, 64,  96, 96]
        c3 = self.stage1(x)    # [B, 128, 48, 48]
        c4 = self.stage2(c3)   # [B, 256, 24, 24]
        c5 = self.stage3(c4)   # [B, 512, 12, 12]
        return {"C3": c3, "C4": c4, "C5": c5}


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = EfficientViTEncoder()
    x = torch.randn(2, 3, 384, 384)
    feats = model(x)
    for k, v in feats.items():
        print(f"{k}: {v.shape}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {n_params:.1f}M")
