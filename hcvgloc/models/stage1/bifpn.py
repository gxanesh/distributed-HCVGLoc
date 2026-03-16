"""
bifpn.py
--------
Bi-directional Feature Pyramid Network (BiFPN) neck for HCVGLoc Stage 1.

Fuses multi-scale features {C3, C4, C5} from EfficientViT-B1 using:
  - Top-down pathway  (C5 → C4 → C3)
  - Bottom-up pathway (C3 → C4 → C5)
  - Learnable weighted fusion at each node

Output: unified 256-D feature maps at three scales.

Reference: EfficientDet: Scalable and Efficient Object Detection
           (Tan et al., CVPR 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSepConv(nn.Module):
    """Depthwise-separable conv BN GELU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class WeightedFusion(nn.Module):
    """
    Fast normalised weighted feature fusion (BiFPN-style).
    out = Σ(wᵢ · xᵢ) / (Σwᵢ + ε)   where wᵢ = ReLU(raw_wᵢ)
    """

    def __init__(self, n_inputs: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_inputs))

    def forward(self, features: list) -> torch.Tensor:
        w = F.relu(self.w)
        denom = w.sum() + 1e-6
        out = sum(w[i] * features[i] for i in range(len(features)))
        return out / denom


class BiFPNLayer(nn.Module):
    """
    Single BiFPN layer performing one full top-down + bottom-up sweep.

    Channels: all unified to `out_ch` (default 256).
    """

    def __init__(self, in_channels: dict, out_ch: int = 256):
        """
        Args:
            in_channels: {'C3': 128, 'C4': 256, 'C5': 512}
            out_ch: unified output channels
        """
        super().__init__()
        self.out_ch = out_ch

        # Lateral projections (input → out_ch)
        self.lat_c3 = nn.Conv2d(in_channels["C3"], out_ch, 1, bias=False)
        self.lat_c4 = nn.Conv2d(in_channels["C4"], out_ch, 1, bias=False)
        self.lat_c5 = nn.Conv2d(in_channels["C5"], out_ch, 1, bias=False)

        # ----- Top-down pathway -----
        # P4_td = fuse(C4, upsample(C5))
        self.td_fuse_c4 = WeightedFusion(2)
        self.td_conv_c4 = DepthwiseSepConv(out_ch, out_ch)

        # P3_td = fuse(C3, upsample(P4_td))
        self.td_fuse_c3 = WeightedFusion(2)
        self.td_conv_c3 = DepthwiseSepConv(out_ch, out_ch)

        # ----- Bottom-up pathway -----
        # P4_out = fuse(C4, P4_td, downsample(P3_td))
        self.bu_fuse_c4 = WeightedFusion(3)
        self.bu_conv_c4 = DepthwiseSepConv(out_ch, out_ch)

        # P5_out = fuse(C5, downsample(P4_out))
        self.bu_fuse_c5 = WeightedFusion(2)
        self.bu_conv_c5 = DepthwiseSepConv(out_ch, out_ch)

    def forward(self, features: dict) -> dict:
        """
        Args:
            features: {'C3': [B,128,48,48], 'C4': [B,256,24,24], 'C5': [B,512,12,12]}
        Returns:
            {'P3': [B,256,48,48], 'P4': [B,256,24,24], 'P5': [B,256,12,12]}
        """
        c3 = self.lat_c3(features["C3"])
        c4 = self.lat_c4(features["C4"])
        c5 = self.lat_c5(features["C5"])

        # Top-down
        p4_td = self.td_conv_c4(self.td_fuse_c4([
            c4,
            F.interpolate(c5, size=c4.shape[-2:], mode="nearest"),
        ]))
        p3_td = self.td_conv_c3(self.td_fuse_c3([
            c3,
            F.interpolate(p4_td, size=c3.shape[-2:], mode="nearest"),
        ]))

        # Bottom-up
        p4_out = self.bu_conv_c4(self.bu_fuse_c4([
            c4,
            p4_td,
            F.adaptive_avg_pool2d(p3_td, c4.shape[-2:]),
        ]))
        p5_out = self.bu_conv_c5(self.bu_fuse_c5([
            c5,
            F.adaptive_avg_pool2d(p4_out, c5.shape[-2:]),
        ]))

        return {"P3": p3_td, "P4": p4_out, "P5": p5_out}


class BiFPN(nn.Module):
    """
    Stack of BiFPN layers.

    Args:
        in_channels: channel sizes from EfficientViT-B1 stages
        out_ch: unified channel width (default 256)
        num_layers: number of BiFPN stacks (default 3)
    """

    def __init__(
        self,
        in_channels: dict = {"C3": 128, "C4": 256, "C5": 512},
        out_ch: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        # First layer: takes raw backbone channels
        layers = [BiFPNLayer(in_channels, out_ch)]
        # Subsequent layers: all inputs are already out_ch
        for _ in range(num_layers - 1):
            layers.append(BiFPNLayer({"C3": out_ch, "C4": out_ch, "C5": out_ch}, out_ch))
        self.layers = nn.ModuleList(layers)

    def forward(self, features: dict) -> dict:
        """features: {'C3', 'C4', 'C5'} from EfficientViT"""
        x = features
        for i, layer in enumerate(self.layers):
            if i > 0:
                # Rename P→C for subsequent layers
                x = {"C3": x["P3"], "C4": x["P4"], "C5": x["P5"]}
            x = layer(x)
        return x   # {'P3', 'P4', 'P5'}
