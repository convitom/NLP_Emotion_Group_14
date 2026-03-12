"""
src/train.py
Two-Stage Ekman Emotion Classification — Training.

Outputs saved to:
    <run_base_dir>/run_2_stage/<s1>+<s2>[_N]/
        checkpoints/stage1_best.pth
        checkpoints/stage2_best.pth
        logs/training_log_stage1.csv
        logs/training_log_stage2.csv

CLI:
    python src/train.py --stage stage1
    python src/train.py --stage stage2
    python src/train.py --stage all          # trains stage1 then stage2
    python src/train.py --config config/config.yaml --stage all

Notebook:
    from src.train import train
    run_dir = train(config_path="config/config.yaml", stage="stage1")
    run_dir = train(config_path="config/config.yaml", stage="stage2", run_dir=run_dir)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
from transformers import AutoModel

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataloader import get_dataloaders, EMOTION_NAMES, NUM_EMOTIONS, BACKBONE_REGISTRY
from src.utils import (
    AverageMeter, get_optimizer, get_run_dir,
    get_scheduler, load_config, set_seed, apply_threshold,
    save_config_summary,
)
from models.loss import get_loss_fn


# =============================================================================
#  Model definitions
# =============================================================================

class EncoderForBinaryClassification(nn.Module):
    """Stage 1: single binary head. Output: (B, 1) logit."""

    def __init__(self, pretrained_name: str, dropout: float = 0.1):
        super().__init__()
        self.backbone   = AutoModel.from_pretrained(pretrained_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.classifier(cls)   # (B, 1)


class EncoderForMultiLabelClassification(nn.Module):
    """Stage 2: independent binary head per class. Output: (B, num_labels) logits."""

    def __init__(self, pretrained_name: str, num_labels: int = NUM_EMOTIONS, dropout: float = 0.1):
        super().__init__()
        self.backbone    = AutoModel.from_pretrained(pretrained_name)
        self.dropout     = nn.Dropout(dropout)
        hidden           = self.backbone.config.hidden_size
        self.classifiers = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(num_labels)])
        self.num_labels  = num_labels

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return torch.cat([h(cls) for h in self.classifiers], dim=1)   # (B, C)


def build_model(stage_cfg: dict, stage: str, num_labels: int = None) -> nn.Module:
    name = stage_cfg["model"]["name"].lower()
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {' | '.join(BACKBONE_REGISTRY)}")
    pretrained = BACKBONE_REGISTRY[name]["pretrained"]
    dropout    = float(stage_cfg["model"].get("dropout", 0.1))
    if stage == "stage1":
        return EncoderForBinaryClassification(pretrained, dropout=dropout)
    n = num_labels if num_labels is not None else NUM_EMOTIONS
    return EncoderForMultiLabelClassification(pretrained, num_labels=n, dropout=dropout)


# =============================================================================
#  One-epoch helper
# =============================================================================

def _run_epoch(
    model, loader, criterion, optimizer, scheduler, scaler,
    device, stage, phase, epoch, total_epochs,
    threshold=0.5, use_amp=False, amp_dtype=torch.float16,
) -> Tuple[float, Dict[str, float]]:
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    loss_meter  = AverageMeter("loss")
    all_preds:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [{phase:>5}]",
                leave=True, dynamic_ncols=True, unit="batch", mininterval=1.0)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in pbar:
            ids   = batch["input_ids"].to(device,      non_blocking=True)
            mask  = batch["attention_mask"].to(device, non_blocking=True)
            lbls  = batch["labels"].to(device,         non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(ids, mask)
                loss   = criterion(logits, lbls)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            loss_meter.update(loss.item(), lbls.size(0))
            if not is_train:
                # Cast to float32 before numpy — bfloat16 is not supported by numpy
                probs = torch.sigmoid(logits).float().cpu().numpy()
                all_preds.append(apply_threshold(probs, threshold))
                all_labels.append(lbls.float().cpu().numpy())

            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")
    pbar.close()

    metrics: Dict[str, float] = {}
    if not is_train and all_preds:
        y_true = np.vstack(all_labels)
        y_pred = np.vstack(all_preds)
        if stage == "stage1":
            yf, pf = y_true.flatten(), y_pred.flatten()
            metrics = {
                "accuracy":  float(accuracy_score(yf, pf)),
                "f1":        float(f1_score(yf, pf, zero_division=0)),
                "precision": float(precision_score(yf, pf, zero_division=0)),
                "recall":    float(recall_score(yf, pf, zero_division=0)),
            }
        else:
            metrics = {
                "micro_f1":    float(f1_score(y_true, y_pred, average="micro",    zero_division=0)),
                "macro_f1":    float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
                "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }

    return loss_meter.avg, metrics


# =============================================================================
#  Main training function
# =============================================================================

def train(
    config_path: str = "config/config.yaml",
    stage:       str = "stage1",
    run_dir:     str = None,
) -> Dict:
    """
    Train one stage.

    Args:
        config_path: path to config YAML.
        stage:       "stage1" or "stage2".
        run_dir:     existing run directory to reuse (e.g. pass stage1's run_dir
                     to stage2 so both share the same output folder).
                     If None, a new run directory is created.

    Returns:
        dict with keys: stage, run_dir, run_name, best_val_loss,
                        best_epoch, best_metrics, log_path, checkpoint_path.
    """
    assert stage in ("stage1", "stage2")

    cfg       = load_config(config_path)
    stage_cfg = cfg[stage]
    train_cfg = stage_cfg["training"]
    set_seed(cfg["data"]["seed"])

    epochs    = int(train_cfg["epochs"])
    patience  = int(train_cfg.get("early_stopping_patience", 3))
    threshold = float(train_cfg.get("threshold", 0.5))

    # ── Device & AMP ──────────────────────────────────────────────────────────
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_key     = stage_cfg["model"]["name"].lower()
    amp_dtype_str = BACKBONE_REGISTRY.get(model_key, {}).get("amp_dtype", "float16")

    if device.type == "cuda" and amp_dtype_str == "bfloat16":
        use_amp, amp_dtype, use_scaler = True, torch.bfloat16, False
    elif device.type == "cuda":
        use_amp, amp_dtype, use_scaler = True, torch.float16, True
    else:
        use_amp, amp_dtype, use_scaler = False, torch.float32, False

    # ── Run directory ─────────────────────────────────────────────────────────
    if run_dir is None:
        run_dir, run_name = get_run_dir(cfg)
        print(f"\n{'='*62}")
        print(f"  New run: {run_name}")
        print(f"  Path   : {run_dir}")
        print(f"{'='*62}")
    else:
        run_name = os.path.basename(run_dir)
        print(f"\n{'='*62}")
        print(f"  Continuing run: {run_name}  (stage={stage})")
        print(f"{'='*62}")

    print(f"[train] Stage={stage}  Device={device}  AMP={use_amp} ({amp_dtype_str})")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _, info = get_dataloaders(cfg, stage=stage)
    pos_weight   = info["pos_weight"]
    num_labels   = info["num_labels"]
    tier_indices = info["tier_indices"]
    pretrained   = info["pretrained"]

    # ── Model ─────────────────────────────────────────────────────────────────
    model      = build_model(stage_cfg, stage=stage, num_labels=num_labels).to(device)
    model_name = stage_cfg["model"]["name"]
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model  : {model_name}  ({pretrained})")
    print(f"[train] Params : {n_params:,}  |  Labels: {num_labels}")

    # ── Save config summary ───────────────────────────────────────────────────
    save_config_summary(
        run_dir=run_dir, cfg=cfg, stage=stage, info=info,
        n_params=n_params,
        n_train=len(train_loader.dataset),
        n_val=len(val_loader.dataset),
        n_test=0,   # test set not loaded during training
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    cfg_for_loss = {"training": dict(train_cfg)}
    criterion    = get_loss_fn(cfg_for_loss, device, pos_weight=pos_weight,
                               tier_indices=tier_indices)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer   = get_optimizer(model, cfg_for_loss)
    total_steps = len(train_loader) * epochs
    scheduler   = get_scheduler(optimizer, cfg_for_loss, num_training_steps=total_steps)
    scaler      = GradScaler() if use_scaler else None

    # ── Paths ─────────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(run_dir, "checkpoints", f"{stage}_best.pth")
    log_path  = os.path.join(run_dir, "logs",        f"training_log_{stage}.csv")

    # ── CSV header ────────────────────────────────────────────────────────────
    if stage == "stage1":
        csv_fields = ["epoch", "train_loss", "val_loss",
                      "val_accuracy", "val_f1", "val_precision", "val_recall", "lr"]
    else:
        csv_fields = ["epoch", "train_loss", "val_loss",
                      "val_micro_f1", "val_macro_f1", "val_weighted_f1", "lr"]

    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_metrics: Dict[str, float] = {}
    best_epoch    = 0
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, _ = _run_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, stage, "train", epoch, epochs,
            threshold, use_amp=use_amp, amp_dtype=amp_dtype,
        )
        val_loss, val_metrics = _run_epoch(
            model, val_loader, criterion, None, None,
            None, device, stage, "val", epoch, epochs,
            threshold, use_amp=use_amp, amp_dtype=amp_dtype,
        )

        lr      = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        if stage == "stage1":
            print(
                f"Epoch {epoch:>3}/{epochs}  "
                f"tr_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"acc={val_metrics.get('accuracy',0):.4f}  "
                f"f1={val_metrics.get('f1',0):.4f}  "
                f"p={val_metrics.get('precision',0):.4f}  r={val_metrics.get('recall',0):.4f}  "
                f"lr={lr:.2e}  [{elapsed:.0f}s]"
            )
            row = {"epoch": epoch, "train_loss": round(train_loss, 6),
                   "val_loss": round(val_loss, 6),
                   "val_accuracy":  round(val_metrics.get("accuracy",  0), 4),
                   "val_f1":        round(val_metrics.get("f1",        0), 4),
                   "val_precision": round(val_metrics.get("precision", 0), 4),
                   "val_recall":    round(val_metrics.get("recall",    0), 4),
                   "lr": lr}
        else:
            print(
                f"Epoch {epoch:>3}/{epochs}  "
                f"tr_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"micro_f1={val_metrics.get('micro_f1',0):.4f}  "
                f"macro_f1={val_metrics.get('macro_f1',0):.4f}  "
                f"w_f1={val_metrics.get('weighted_f1',0):.4f}  "
                f"lr={lr:.2e}  [{elapsed:.0f}s]"
            )
            row = {"epoch": epoch, "train_loss": round(train_loss, 6),
                   "val_loss": round(val_loss, 6),
                   "val_micro_f1":    round(val_metrics.get("micro_f1",    0), 4),
                   "val_macro_f1":    round(val_metrics.get("macro_f1",    0), 4),
                   "val_weighted_f1": round(val_metrics.get("weighted_f1", 0), 4),
                   "lr": lr}

        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics  = val_metrics.copy()
            best_epoch    = epoch
            no_improve    = 0
            torch.save({
                "stage":           stage,
                "epoch":           epoch,
                "model_name":      model_name,
                "pretrained_name": pretrained,
                "num_labels":      num_labels,
                "tier_indices":    tier_indices,
                "model_state":     model.state_dict(),
                "optimizer":       optimizer.state_dict(),
                "scheduler":       scheduler.state_dict() if scheduler else None,
                "val_loss":        best_val_loss,
                "val_metrics":     best_metrics,
                "threshold":       threshold,
                "cfg":             cfg,
                "run_dir":         run_dir,
                "run_name":        run_name,
            }, ckpt_path)
            print(f"  ✓ Checkpoint saved → {ckpt_path}")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[train] Early stopping at epoch {epoch} ({patience} epochs no improve).")
            break

    print(f"\n[train] {stage.upper()} done — best val_loss={best_val_loss:.4f} @ epoch {best_epoch}")
    for k, v in best_metrics.items():
        print(f"         {k}={v:.4f}")
    print(f"[train] log → {log_path}")

    return {
        "stage":            stage,
        "run_dir":          run_dir,
        "run_name":         run_name,
        "best_val_loss":    best_val_loss,
        "best_epoch":       best_epoch,
        "best_metrics":     best_metrics,
        "log_path":         log_path,
        "checkpoint_path":  ckpt_path,
    }


# =============================================================================
#  CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",  type=str, default="all",
                        choices=["stage1", "stage2", "all"])
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    if args.stage == "all":
        r1 = train(config_path=args.config, stage="stage1")
        train(config_path=args.config, stage="stage2", run_dir=r1["run_dir"])
    else:
        train(config_path=args.config, stage=args.stage)
