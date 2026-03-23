from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["BCELoss", "FocalBCELoss", "AsymmetricLoss", "TieredPerClassASL", "get_loss_fn"]
class BCELoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)

class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 pos_weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw  = getattr(self, "pos_weight", None)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        loss = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * \
               ((1 - p_t) ** self.gamma) * bce
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss

class AsymmetricLoss(nn.Module):

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
                 clip: float = 0.05, reduction: str = "mean"):
        super().__init__()
        self.gamma_pos, self.gamma_neg, self.clip, self.reduction = gamma_pos, gamma_neg, clip, reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p    = torch.sigmoid(logits)
        pn   = (p - self.clip).clamp(min=1e-8, max=1.0 - 1e-8)
        p_s  = p.clamp(min=1e-8, max=1.0 - 1e-8)
        lp   = targets       * torch.log(p_s)
        ln   = (1 - targets) * torch.log(1.0 - pn)
        if self.gamma_pos > 0: lp = ((1.0 - p_s) ** self.gamma_pos) * lp
        if self.gamma_neg > 0: ln = (pn           ** self.gamma_neg) * ln
        loss = -(lp + ln)
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss

class TieredPerClassASL(nn.Module):
    def __init__(
        self,
        tier_indices:           Dict[str, List[int]],
        gamma_pos_very_rare:    float = 0.0,
        gamma_neg_very_rare:    float = 0.5,
        clip_very_rare:         float = 0.0,
        gamma_pos_rare:         float = 0.5,
        gamma_neg_rare:         float = 1.0,
        clip_rare:              float = 0.0,
        gamma_pos_common:       float = 1.0,
        gamma_neg_common:       float = 2.0,
        clip_common:            float = 0.0,
        reduction:              str   = "mean",
    ) -> None:
        super().__init__()
        self.tier_indices = tier_indices
        self.params = {
            "very_rare": (gamma_pos_very_rare, gamma_neg_very_rare, clip_very_rare),
            "rare":      (gamma_pos_rare,      gamma_neg_rare,      clip_rare),
            "common":    (gamma_pos_common,    gamma_neg_common,    clip_common),
        }
        self.reduction = reduction
        self._num_classes = sum(len(v) for v in tier_indices.values())

    def _get_tier(self, c: int) -> str:
        if c in self.tier_indices.get("very_rare", []):
            return "very_rare"
        if c in self.tier_indices.get("rare", []):
            return "rare"
        return "common"

    def _asl_col(self, p: torch.Tensor, t: torch.Tensor,
                  gp: float, gn: float, clip: float) -> torch.Tensor:
        pn  = (p - clip).clamp(min=1e-8, max=1.0 - 1e-8)
        p_s = p.clamp(min=1e-8, max=1.0 - 1e-8)   
        lp  = t       * torch.log(p_s)
        ln  = (1 - t) * torch.log(1.0 - pn)
        if gp > 0: lp = ((1.0 - p_s) ** gp) * lp
        if gn > 0: ln = (pn           ** gn) * ln
        return -(lp + ln)   

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p    = torch.sigmoid(logits)
        C    = logits.size(1)
        cols = []
        for c in range(C):
            tier       = self._get_tier(c)
            gp, gn, cl = self.params[tier]
            cols.append(self._asl_col(p[:, c], targets[:, c], gp, gn, cl).unsqueeze(1))
        loss = torch.cat(cols, dim=1)   
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss

def get_loss_fn(
    cfg:         dict,
    device:      torch.device,
    pos_weight:  Optional[torch.Tensor] = None,
    tier_indices: Optional[Dict[str, List[int]]] = None,
) -> nn.Module:
  
    t     = cfg.get("training", {})
    name  = t.get("loss", "bce_weighted").lower().strip()

    if name == "bce":
        return BCELoss().to(device)

    if name == "bce_weighted":
        if pos_weight is None:
            raise ValueError("bce_weighted requires pos_weight")
        return BCELoss(pos_weight=pos_weight.to(device)).to(device)

    if name == "focal_bce":
        pw = pos_weight.to(device) if pos_weight is not None else None
        return FocalBCELoss(gamma=float(t.get("focal_gamma", 2.0)), pos_weight=pw).to(device)

    if name == "asymmetric":
        return AsymmetricLoss(
            gamma_pos=float(t.get("asl_gamma_pos", 0.0)),
            gamma_neg=float(t.get("asl_gamma_neg", 4.0)),
            clip=     float(t.get("asl_clip",      0.05)),
        ).to(device)

    if name == "per_class_asl":
        if tier_indices is None:
            raise ValueError("per_class_asl requires tier_indices")
        return TieredPerClassASL(
            tier_indices=tier_indices,
            gamma_pos_very_rare=float(t.get("asl_gamma_pos_very_rare", 0.0)),
            gamma_neg_very_rare=float(t.get("asl_gamma_neg_very_rare", 0.5)),
            clip_very_rare=     float(t.get("asl_clip_very_rare",      0.0)),
            gamma_pos_rare=     float(t.get("asl_gamma_pos_rare",      0.5)),
            gamma_neg_rare=     float(t.get("asl_gamma_neg_rare",      1.0)),
            clip_rare=          float(t.get("asl_clip_rare",           0.0)),
            gamma_pos_common=   float(t.get("asl_gamma_pos_common",    1.0)),
            gamma_neg_common=   float(t.get("asl_gamma_neg_common",    2.0)),
            clip_common=        float(t.get("asl_clip_common",         0.0)),
        ).to(device)

    raise ValueError(
        f"Unknown loss '{name}'. "
        "Choose from: bce | bce_weighted | focal_bce | asymmetric | per_class_asl"
    )
