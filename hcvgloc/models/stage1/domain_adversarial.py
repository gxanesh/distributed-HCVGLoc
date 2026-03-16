"""
domain_adversarial.py
----------------------
Gradient Reversal Layer (GRL) + Domain Classifier for HCVGLoc Stage 1.

Purpose: Force the backbone to learn domain-invariant features so that
descriptors generalise across geographic areas (cross-area generalisation
gap: 20-35% R@1 drop without this module).

Mechanism:
    - Forward pass:  identity (passes features through unchanged)
    - Backward pass: multiplies gradient by -λ (reverses gradient)
    → Backbone is simultaneously trained to:
        (a) produce discriminative embeddings (via InfoNCE loss)
        (b) confuse the domain classifier (via reversed gradient)

λ scheduling: linearly ramp from 0 → λ_max over warmup_epochs to
              prevent gradient explosion in early training.

Reference: Domain-Adversarial Training of Neural Networks
           (Ganin et al., JMLR 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Gradient Reversal Function
# ---------------------------------------------------------------------------

class GradientReversalFunction(Function):
    """Custom autograd function implementing gradient reversal."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        lambda_ = ctx.saved_tensors[0].item()
        return -lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps GradientReversalFunction as a standard nn.Module."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


# ---------------------------------------------------------------------------
# Domain Classifier
# ---------------------------------------------------------------------------

class DomainClassifier(nn.Module):
    """
    3-layer MLP domain classifier placed after GRL.

    Domains (4):
        0 → CVUSA
        1 → VIGOR
        2 → CVACT
        3 → University-1652

    Args:
        input_dim: descriptor dimension (512)
        hidden_dim: hidden layer width
        num_domains: number of dataset domains
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_domains: int = 4,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]  — L2-normalised descriptor
        Returns:
            logits: [B, num_domains]
        """
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Combined GRL + Classifier module
# ---------------------------------------------------------------------------

class DomainAdversarialModule(nn.Module):
    """
    Full domain adversarial module:
        descriptor → GRL → DomainClassifier → domain logits

    λ is scheduled externally via set_lambda().
    """

    def __init__(
        self,
        descriptor_dim: int = 512,
        hidden_dim: int = 256,
        num_domains: int = 4,
        lambda_init: float = 0.0,
    ):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=lambda_init)
        self.classifier = DomainClassifier(descriptor_dim, hidden_dim, num_domains)

    def set_lambda(self, lambda_: float):
        """Called by the trainer at each epoch to schedule λ."""
        self.grl.set_lambda(lambda_)

    @staticmethod
    def schedule_lambda(
        epoch: int,
        warmup_epochs: int = 20,
        lambda_max: float = 0.1,
    ) -> float:
        """
        Linear ramp from 0 → lambda_max over warmup_epochs.

        Args:
            epoch: current training epoch (0-indexed)
            warmup_epochs: epochs over which to ramp λ
            lambda_max: maximum λ value

        Returns:
            λ value for this epoch
        """
        if epoch < warmup_epochs:
            return lambda_max * (epoch / warmup_epochs)
        return lambda_max

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            descriptor: [B, descriptor_dim]
        Returns:
            domain_logits: [B, num_domains]
        """
        reversed_feat = self.grl(descriptor)
        return self.classifier(reversed_feat)
