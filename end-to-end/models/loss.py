"""
models/loss.py
Loss functions for Ekman Emotion Classification.

Supported loss types (config key: training.loss):
    "bce"           Standard BCEWithLogitsLoss (no pos_weight)
    "bce_weighted"  BCEWithLogitsLoss with inverse-frequency pos_weight
    "asymmetric"    Asymmetric Loss (ASL) with pos_weight — recommended for
                    multi-label classification (gamma_pos=0, gamma_neg=2, clip=0.05)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["BCELoss", "AsymmetricLoss", "get_loss_fn"]


# =============================================================================
#  1. BCE  (standard and weighted)
# =============================================================================

class BCELoss(nn.Module):
    """
    Wrapper around BCEWithLogitsLoss.
    Pass pos_weight to correct for class-frequency imbalance.
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction:  str = "mean",
    ) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


# =============================================================================
#  2. Asymmetric Loss  (Ridnik et al., 2021)
# =============================================================================

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.

    Applies independent focusing exponents to positive and negative examples:

        p_m  = max(p - clip, 0)                     (shifted negative probability)
        L_c  = -[ w_c · (1-p)^γ⁺ · t · log(p)
                  + p_m^γ⁻ · (1-t) · log(1-p_m) ]

    Key design choices vs. standard Focal Loss:
        - γ⁺ = 0  → positive-side loss is plain weighted BCE; full gradient
                    preserved for positive examples, which are already scarce.
        - γ⁻ = 2  → negative-side loss is down-weighted by p², suppressing
                    easy negative examples (p ≈ 0) that dominate the gradient
                    in multi-label settings.
        - clip    → shifts the negative probability down by a small margin
                    before applying the focusing factor, further suppressing
                    very confident negative predictions.

    This formulation separates the concern of class-frequency imbalance
    (handled by pos_weight) from easy-sample dominance (handled by γ⁻).

    Args:
        gamma_pos:  Focusing exponent for positive examples (default 0).
        gamma_neg:  Focusing exponent for negative examples (default 2).
        clip:       Probability margin for negative shifting (default 0.05).
        pos_weight: (C,) tensor of per-class positive weights (w_c = n_neg/n_pos).
        reduction:  'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        gamma_pos:  float = 0.0,
        gamma_neg:  float = 2.0,
        clip:       float = 0.05,
        pos_weight: Optional[torch.Tensor] = None,
        reduction:  str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip      = clip
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p   = torch.sigmoid(logits)
        p_s = p.clamp(min=1e-8, max=1.0 - 1e-8)      # safe p for log / power

        # Positive side: (1-p)^γ⁺ · t · log(p)
        lp = targets * torch.log(p_s)
        if self.gamma_pos > 0:
            lp = ((1.0 - p_s) ** self.gamma_pos) * lp

        # Optional per-class weighting on the positive side
        pw = getattr(self, "pos_weight", None)
        if pw is not None:
            lp = pw * lp

        # Negative side: p_m^γ⁻ · (1-t) · log(1-p_m)
        p_m = (p - self.clip).clamp(min=1e-8, max=1.0 - 1e-8)
        ln  = (1.0 - targets) * torch.log(1.0 - p_m)
        if self.gamma_neg > 0:
            ln = (p_m ** self.gamma_neg) * ln

        loss = -(lp + ln)   # (N, C)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# =============================================================================
#  Factory
# =============================================================================

def get_loss_fn(
    cfg:        dict,
    device:     torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Build and return the configured loss function.

    Args:
        cfg:        Config dict containing a "training" sub-dict.
        device:     Target device.
        pos_weight: (C,) tensor of per-class positive weights.
                    Required for "bce_weighted" and "asymmetric".

    Raises:
        ValueError: If an unknown loss name is specified or pos_weight is
                    missing when required.
    """
    t    = cfg.get("training", {})
    name = t.get("loss", "asymmetric").lower().strip()

    if name == "bce":
        return BCELoss().to(device)

    if name == "bce_weighted":
        if pos_weight is None:
            raise ValueError("'bce_weighted' requires pos_weight.")
        return BCELoss(pos_weight=pos_weight.to(device)).to(device)

    if name == "asymmetric":
        pw = pos_weight.to(device) if pos_weight is not None else None
        return AsymmetricLoss(
            gamma_pos=float(t.get("asl_gamma_pos", 0.0)),
            gamma_neg=float(t.get("asl_gamma_neg", 2.0)),
            clip=     float(t.get("asl_clip",      0.05)),
            pos_weight=pw,
        ).to(device)

    raise ValueError(
        f"Unknown loss '{name}'. Choose from: bce | bce_weighted | asymmetric"
    )
