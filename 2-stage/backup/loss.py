"""
models/loss.py
Loss functions for Two-Stage Ekman Emotion Classification.

Supported loss types (config key training.loss):
  "bce"           Standard BCEWithLogitsLoss
  "bce_weighted"  BCEWithLogitsLoss with pos_weight        ← Stage 1 default
  "focal_bce"     Focal-weighted BCE
  "asymmetric"    ASL — same gamma for all classes
  "per_class_asl" Three-tier ASL — very_rare / rare / common   ← Stage 2 default

NaN-safety fixes applied:
  - All sigmoid outputs clamped to [eps, 1-eps] before log
  - pn clamped to [0, 1-eps] to prevent log(0) when clip=0
  - pos_weight clamped to [0.01, 200] as a sanity guard
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BCELoss", "FocalBCELoss", "AsymmetricLoss", "TieredPerClassASL", "get_loss_fn"]

_EPS = 1e-7

#  1. BCE
class BCELoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        if pos_weight is not None:
            pos_weight = pos_weight.clamp(min=0.01, max=200.0)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)

#  2. Focal BCE
class FocalBCELoss(nn.Module):
    def __init__(
        self,
        gamma:      float = 2.0,
        alpha:      float = 0.25,
        pos_weight: Optional[torch.Tensor] = None,
        reduction:  str = "mean",
    ):
        super().__init__()
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.clamp(min=0.01, max=200.0))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw  = getattr(self, "pos_weight", None)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
        p   = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)           # prob of true class
        focal_weight = (self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)) * \
                       ((1.0 - p_t).clamp(min=0.0) ** self.gamma)
        loss = focal_weight * bce
        return loss.mean() if self.reduction == "mean" else \
               loss.sum()  if self.reduction == "sum"  else loss



#  3. Standard ASL

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss — same gamma/clip for all classes."""

    def __init__(
        self,
        gamma_pos:  float = 0.0,
        gamma_neg:  float = 4.0,
        clip:       float = 0.05,
        reduction:  str   = "mean",
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip      = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.clamp(min=-50.0, max=50.0)  
        p  = torch.sigmoid(logits)

        # Positive branch — clamp p away from 0
        p_safe = p.clamp(min=_EPS, max=1.0-_EPS)
        lp = targets * torch.log(p_safe)
        if self.gamma_pos > 0:
            lp = ((1.0 - p) ** self.gamma_pos) * lp

        # Negative branch — shift & clamp pn away from 1
        pn = (p - self.clip).clamp(min=0.0, max=1.0 - _EPS)
        ln = (1.0 - targets) * torch.log((1.0 - pn).clamp(min=_EPS))
        if self.gamma_neg > 0:
            ln = (pn ** self.gamma_neg) * ln

        loss = -(lp + ln)
        return loss.mean() if self.reduction == "mean" else \
               loss.sum()  if self.reduction == "sum"  else loss



#  4. Three-Tier Per-Class ASL  (Stage 2 default)
class TieredPerClassASL(nn.Module):
    """
    Asymmetric Loss with three independent gamma/clip sets per imbalance tier:

      very_rare  (e.g. fear)    — stay sensitive, low gamma_neg
      rare       (e.g. disgust) — moderate gamma_neg
      common     (joy, anger …) — aggressive gamma_neg to down-weight easy negatives

    Distribution for reference:
      joy      19794  common
      anger     7813  common
      surprise  5418  common
      sadness   5392  common
      neutral  14497  common   (E2E task only)
      disgust   3366  rare
      fear      1941  very_rare

    Args:
        tier_indices: {"very_rare": [idx,...], "rare": [idx,...], "common": [idx,...]}
    """

    def __init__(
        self,
        tier_indices:        Dict[str, List[int]],
        # very_rare
        gamma_pos_very_rare: float = 0.0,
        gamma_neg_very_rare: float = 0.5,
        clip_very_rare:      float = 0.0,
        # rare
        gamma_pos_rare:      float = 0.0,
        gamma_neg_rare:      float = 1.0,
        clip_rare:           float = 0.05,
        # common
        gamma_pos_common:    float = 1.0,
        gamma_neg_common:    float = 2.0,
        clip_common:         float = 0.05,
        reduction:           str   = "mean",
    ) -> None:
        super().__init__()
        self.tier_indices = tier_indices
        self.params: Dict[str, tuple] = {
            "very_rare": (gamma_pos_very_rare, gamma_neg_very_rare, clip_very_rare),
            "rare":      (gamma_pos_rare,      gamma_neg_rare,      clip_rare),
            "common":    (gamma_pos_common,    gamma_neg_common,    clip_common),
        }
        self.reduction    = reduction
        self._num_classes = sum(len(v) for v in tier_indices.values())

    def _get_tier(self, c: int) -> str:
        if c in self.tier_indices.get("very_rare", []):
            return "very_rare"
        if c in self.tier_indices.get("rare", []):
            return "rare"
        return "common"

    def _asl_col(
        self,
        p: torch.Tensor,   # (B,) sigmoid output
        t: torch.Tensor,   # (B,) binary target
        gp: float,
        gn: float,
        clip: float,
    ) -> torch.Tensor:
        # Positive branch
        p_safe = p.clamp(min=_EPS)
        lp = t * torch.log(p_safe)
        if gp > 0:
            lp = ((1.0 - p) ** gp) * lp

        # Negative branch — pn must stay < 1 to avoid log(0)
        pn = (p - clip).clamp(min=0.0, max=1.0 - _EPS)
        ln = (1.0 - t) * torch.log((1.0 - pn).clamp(min=_EPS))
        if gn > 0:
            ln = (pn ** gn) * ln

        return -(lp + ln)   # (B,)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p    = torch.sigmoid(logits)            # (B, C)
        cols = []
        for c in range(logits.size(1)):
            gp, gn, cl = self.params[self._get_tier(c)]
            cols.append(
                self._asl_col(p[:, c], targets[:, c], gp, gn, cl).unsqueeze(1)
            )
        loss = torch.cat(cols, dim=1)           # (B, C)
        return loss.mean() if self.reduction == "mean" else \
               loss.sum()  if self.reduction == "sum"  else loss



#  5. Factory
def get_loss_fn(
    cfg:          dict,
    device:       torch.device,
    pos_weight:   Optional[torch.Tensor] = None,
    tier_indices: Optional[Dict[str, List[int]]] = None,
) -> nn.Module:
    """
    Build and return the loss function specified in cfg['training']['loss'].

    Args:
        cfg:          Config dict with a 'training' key.
        device:       Target device.
        pos_weight:   (C,) tensor — required for bce_weighted / focal_bce.
        tier_indices: tier dict   — required for per_class_asl.
    """
    t    = cfg.get("training", {})
    name = t.get("loss", "bce_weighted").lower().strip()

    # Clamp pos_weight here as a global safety net
    if pos_weight is not None:
        pos_weight = pos_weight.to(device).clamp(min=0.01, max=200.0)

    if name == "bce":
        return BCELoss().to(device)

    if name == "bce_weighted":
        if pos_weight is None:
            raise ValueError("loss='bce_weighted' requires pos_weight")
        return BCELoss(pos_weight=pos_weight).to(device)

    if name == "focal_bce":
        return FocalBCELoss(
            gamma=float(t.get("focal_gamma", 2.0)),
            pos_weight=pos_weight,
        ).to(device)

    if name == "asymmetric":
        return AsymmetricLoss(
            gamma_pos=float(t.get("asl_gamma_pos", 0.0)),
            gamma_neg=float(t.get("asl_gamma_neg", 4.0)),
            clip=     float(t.get("asl_clip",      0.05)),
        ).to(device)

    if name == "per_class_asl":
        if tier_indices is None:
            raise ValueError("loss='per_class_asl' requires tier_indices")
        return TieredPerClassASL(
            tier_indices=tier_indices,
            gamma_pos_very_rare=float(t.get("asl_gamma_pos_very_rare", 0.0)),
            gamma_neg_very_rare=float(t.get("asl_gamma_neg_very_rare", 0.5)),
            clip_very_rare=     float(t.get("asl_clip_very_rare",      0.0)),
            gamma_pos_rare=     float(t.get("asl_gamma_pos_rare",      0.0)),
            gamma_neg_rare=     float(t.get("asl_gamma_neg_rare",      1.0)),
            clip_rare=          float(t.get("asl_clip_rare",           0.05)),
            gamma_pos_common=   float(t.get("asl_gamma_pos_common",    1.0)),
            gamma_neg_common=   float(t.get("asl_gamma_neg_common",    2.0)),
            clip_common=        float(t.get("asl_clip_common",         0.05)),
        ).to(device)

    raise ValueError(
        f"Unknown loss '{name}'. "
        "Choose from: bce | bce_weighted | focal_bce | asymmetric | per_class_asl"
    )
