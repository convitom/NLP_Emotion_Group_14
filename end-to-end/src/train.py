"""
src/train.py
Single-Stage End-to-End Ekman Classification — Training.

Task   : 7-class multi-label (6 emotions + neutral), ALL samples.
Model  : EncoderForClassification with 7 independent output heads.
Outputs: <run_base_dir>/run_e2e/<model_name>[_N]/

Output structure:
    <run_base_dir>/run_e2e/<model>/
        checkpoints/best.pth
        logs/training_log.csv
        logs/config.txt
        results/                ← filled by test.py

CLI:
    python src/train.py
    python src/train.py --config config/config.yaml
    python src/train.py --run_dir /content/drive/MyDrive/run_e2e/deberta

Notebook:
    from src.train import train
    result = train(config_path="config/config.yaml")
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
from transformers import AutoModel

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataloader import (
    get_dataloaders,
    CLASS_NAMES, NUM_CLASSES,
    BACKBONE_REGISTRY,
)
from src.utils import (
    AverageMeter, get_optimizer, get_scheduler,
    load_config, set_seed, apply_threshold,
)
from models.loss import get_loss_fn


# =============================================================================
#  Model
# =============================================================================

class EncoderForClassification(nn.Module):
    """7-head multi-label classifier (6 emotions + neutral)."""

    def __init__(self, pretrained_name: str, num_labels: int = NUM_CLASSES,
                 dropout: float = 0.1, from_scratch: bool = False):
        super().__init__()
        if from_scratch:
            from transformers import AutoConfig
            config       = AutoConfig.from_pretrained(pretrained_name)
            self.backbone = AutoModel.from_config(config)
        else:
            self.backbone = AutoModel.from_pretrained(pretrained_name)
        self.dropout     = nn.Dropout(dropout)
        hidden           = self.backbone.config.hidden_size
        self.classifiers = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(num_labels)])
        self.num_labels  = num_labels

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return torch.cat([h(cls) for h in self.classifiers], dim=1)   # (B, 7)


def _apply_freeze_backbone(model: nn.Module) -> None:
    """Freeze tất cả tham số backbone, chỉ để head trainable."""
    for name, param in model.named_parameters():
        if "classifiers" not in name:
            param.requires_grad = False


def build_model(stage_cfg: dict, num_labels: int = NUM_CLASSES) -> nn.Module:
    name = stage_cfg["model"]["name"].lower()
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {' | '.join(BACKBONE_REGISTRY)}")
    pretrained    = BACKBONE_REGISTRY[name]["pretrained"]
    dropout       = float(stage_cfg["model"].get("dropout", 0.1))
    training_mode = stage_cfg["training"].get("training_mode", "finetune").lower().strip()

    valid_modes = {"finetune", "freeze_backbone", "from_scratch", "llrd"}
    if training_mode not in valid_modes:
        raise ValueError(
            f"Unknown training_mode '{training_mode}'. "
            f"Choose from: {' | '.join(sorted(valid_modes))}"
        )

    from_scratch = (training_mode == "from_scratch")
    model = EncoderForClassification(
        pretrained, num_labels=num_labels, dropout=dropout, from_scratch=from_scratch
    )

    if training_mode == "freeze_backbone":
        _apply_freeze_backbone(model)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[build_model] freeze_backbone — frozen={frozen:,}  trainable={trainable:,}")

    elif training_mode == "from_scratch":
        print(f"[build_model] from_scratch — backbone initialized with RANDOM weights (no pretrained)")

    elif training_mode == "llrd":
        print(f"[build_model] llrd — Layerwise LR Decay will be applied by optimizer")

    else:
        print(f"[build_model] finetune — full backbone fine-tuning")

    return model


# =============================================================================
#  Run directory  (YOLO-style)
# =============================================================================

def get_run_dir(cfg: dict) -> Tuple[str, str]:
    base       = cfg.get("run_base_dir", ".")
    model_name = cfg["e2e"]["model"]["name"]
    root       = os.path.join(base, "run_e2e")
    os.makedirs(root, exist_ok=True)

    candidate = os.path.join(root, model_name)
    if not os.path.exists(candidate):
        run_dir, run_name = candidate, model_name
    else:
        n = 2
        while True:
            name = f"{model_name}_{n}"
            candidate = os.path.join(root, name)
            if not os.path.exists(candidate):
                run_dir, run_name = candidate, name
                break
            n += 1

    for sub in ["checkpoints", "logs", "results"]:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    return run_dir, run_name


def get_existing_run_dir(cfg: dict) -> str:
    base       = cfg.get("run_base_dir", ".")
    model_name = cfg["e2e"]["model"]["name"]
    root       = os.path.join(base, "run_e2e")

    candidates = []
    if os.path.isdir(os.path.join(root, model_name)):
        candidates.append((1, os.path.join(root, model_name)))
    n = 2
    while True:
        p = os.path.join(root, f"{model_name}_{n}")
        if os.path.isdir(p):
            candidates.append((n, p))
            n += 1
        else:
            break

    if not candidates:
        raise FileNotFoundError(
            f"No run_e2e directory found for '{model_name}' under '{root}'. "
            "Run train.py first."
        )
    _, latest = sorted(candidates)[-1]
    return latest


# =============================================================================
#  One-epoch helper
# =============================================================================

def _run_epoch(
    model, loader, criterion, optimizer, scheduler, scaler,
    device, phase, epoch, total_epochs,
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
            ids  = batch["input_ids"].to(device,      non_blocking=True)
            mask = batch["attention_mask"].to(device,  non_blocking=True)
            lbls = batch["labels"].to(device,          non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(ids, mask)
                loss   = criterion(logits, lbls)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"\nNaN/Inf gradient (norm={grad_norm:.4f}) — skipping batch, resetting scaler. Đây là tính năng, ko phải bug :)")
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()  # scaler tự giảm scale factor, KHÔNG step optimizer
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    clip = 0.3 if amp_dtype == torch.bfloat16 else 1.0
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"\n[WARNING] NaN/Inf gradient — skipping batch")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            loss_meter.update(loss.item(), lbls.size(0))
            if not is_train:
                probs = torch.sigmoid(logits).float().cpu().numpy()
                all_preds.append(apply_threshold(probs, threshold))
                all_labels.append(lbls.float().cpu().numpy())

            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")
    pbar.close()

    metrics: Dict[str, float] = {}
    if not is_train and all_preds:
        y_true = np.vstack(all_labels)
        y_pred = np.vstack(all_preds)
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
    run_dir:     str = None,
) -> Dict:
    """
    Train single-stage 7-class model.

    Returns:
        dict with keys: run_dir, run_name, best_val_macro_f1,
                        best_epoch, best_metrics, log_path, checkpoint_path.
    """
    cfg       = load_config(config_path)
    stage_cfg = cfg["e2e"]
    train_cfg = stage_cfg["training"]
    set_seed(cfg["data"]["seed"])

    epochs    = int(train_cfg["epochs"])
    patience  = int(train_cfg.get("early_stopping_patience", 3))
    threshold = float(train_cfg.get("threshold", 0.5))

    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_key     = stage_cfg["model"]["name"].lower()
    amp_dtype_str = BACKBONE_REGISTRY.get(model_key, {}).get("amp_dtype", "float16")

    if device.type == "cuda" and amp_dtype_str == "bfloat16":
        use_amp    = True
        amp_dtype  = torch.bfloat16
        use_scaler = False 
    elif device.type == "cuda":
        use_amp    = True
        amp_dtype  = torch.float16
        use_scaler = True
    else:
        use_amp    = False
        amp_dtype  = torch.float32
        use_scaler = False

    # ── Run directory ─────────────────────────────────────────────────────────
    if run_dir is None:
        run_dir, run_name = get_run_dir(cfg)
        print(f"\n{'='*62}\n  New run  : {run_name}\n  Path     : {run_dir}\n{'='*62}")
    else:
        run_name = os.path.basename(run_dir)
        print(f"\n{'='*62}\n  Resuming : {run_name}\n{'='*62}")

    print(f"[train] Device={device}  AMP={use_amp} ({amp_dtype_str})")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _, info = get_dataloaders(cfg)
    pos_weight   = info["pos_weight"]
    num_labels   = info["num_labels"]
    tier_indices = info["tier_indices"]
    pretrained   = info["pretrained"]

    # ── Model ─────────────────────────────────────────────────────────────────
    model      = build_model(stage_cfg, num_labels=num_labels).to(device)
    model_name = stage_cfg["model"]["name"]
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model  : {model_name}  ({pretrained})")
    print(f"[train] Params : {n_params:,}  |  Labels: {num_labels}")

    # ── Loss / Optimizer / Scheduler ──────────────────────────────────────────
    cfg_for_loss = {"training": dict(train_cfg)}
    criterion    = get_loss_fn(cfg_for_loss, device, pos_weight=pos_weight,
                               tier_indices=tier_indices)
    optimizer    = get_optimizer(model, cfg_for_loss)
    total_steps  = len(train_loader) * epochs
    scheduler    = get_scheduler(optimizer, cfg_for_loss, num_training_steps=total_steps)
    scaler       = GradScaler() if use_scaler else None

    # ── Paths ─────────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pth")
    log_path  = os.path.join(run_dir, "logs",        "training_log.csv")

    csv_fields = ["epoch", "train_loss", "val_loss",
                  "val_micro_f1", "val_macro_f1", "val_weighted_f1", "lr"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    _save_config_summary(run_dir, cfg, info, n_params,
                         len(train_loader.dataset), len(val_loader.dataset))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_macro_f1 = -1.0
    best_metrics: Dict[str, float] = {}
    best_epoch    = 0
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, _ = _run_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, "train", epoch, epochs,
            threshold, use_amp=use_amp, amp_dtype=amp_dtype,
        )
        val_loss, val_metrics = _run_epoch(
            model, val_loader, criterion, None, None,
            None, device, "val", epoch, epochs,
            threshold, use_amp=use_amp, amp_dtype=amp_dtype,
        )

        lr      = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

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

        if val_metrics.get("macro_f1", 0) > best_val_macro_f1:
            best_val_macro_f1 = val_metrics.get("macro_f1", 0)
            best_metrics  = val_metrics.copy()
            best_epoch    = epoch
            no_improve    = 0
            torch.save({
                "epoch":           epoch,
                "model_name":      model_name,
                "pretrained_name": pretrained,
                "num_labels":      num_labels,
                "tier_indices":    tier_indices,
                "model_state":     model.state_dict(),
                "val_macro_f1":    best_val_macro_f1,
                "val_metrics":     best_metrics,
                "threshold":       threshold,
            }, ckpt_path)
            print(f"  ✓ Checkpoint saved → {ckpt_path}")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[train] Early stopping at epoch {epoch} ({patience} epochs no improve).")
            break

    print(f"\n[train] Done — best val_macro_f1={best_val_macro_f1:.4f} @ epoch {best_epoch}")
    for k, v in best_metrics.items():
        print(f"        {k}={v:.4f}")
    print(f"[train] log → {log_path}")

    return {
        "run_dir":            run_dir,
        "run_name":           run_name,
        "best_val_macro_f1":  best_val_macro_f1,
        "best_epoch":         best_epoch,
        "best_metrics":       best_metrics,
        "log_path":           log_path,
        "checkpoint_path":    ckpt_path,
    }


# =============================================================================
#  Config summary
# =============================================================================

def _save_config_summary(run_dir, cfg, info, n_params, n_train, n_val):
    stage_cfg     = cfg["e2e"]
    train_cfg     = stage_cfg["training"]
    data_cfg      = cfg["data"]
    model_name    = stage_cfg["model"]["name"]
    pretrained    = BACKBONE_REGISTRY.get(model_name, {}).get("pretrained", "?")
    tier_indices  = info.get("tier_indices", {})
    label_counts  = info.get("label_counts", {})

    lines = []
    A = lines.append
    A("=" * 60)
    A("  Config Summary  —  E2E Single-Stage (7-class)")
    A(f"  Run : {os.path.basename(run_dir)}")
    A(f"  Path: {run_dir}")
    A("=" * 60)
    A("\n[Model]")
    A(f"  name          : {model_name}")
    A(f"  pretrained    : {pretrained}")
    A(f"  dropout       : {stage_cfg['model'].get('dropout', 0.1)}")
    A(f"  params        : {n_params:,}")
    A(f"  num_labels    : {info.get('num_labels', NUM_CLASSES)}")
    A(f"  training_mode : {stage_cfg['training'].get('training_mode', 'finetune')}")
    if stage_cfg['training'].get('training_mode', 'finetune') == 'llrd':
        A(f"  llrd_decay    : {stage_cfg['training'].get('llrd_decay_rate', 0.8)}")
    A("\n[Data]")
    A(f"  train_file  : {data_cfg.get('train_file', '?')}")
    A(f"  auto_split  : {data_cfg.get('auto_split', True)}")
    A(f"  max_length  : {data_cfg.get('max_length', 128)}")
    A(f"  num_workers : {data_cfg.get('num_workers', 2)}")
    A(f"  n_train     : {n_train}")
    A(f"  n_val       : {n_val}")
    A("\n[Class Counts & Tiers]")
    for i, name in enumerate(CLASS_NAMES):
        c    = label_counts.get(name, 0)
        tier = ("very_rare" if i in tier_indices.get("very_rare", []) else
                "rare"      if i in tier_indices.get("rare",      []) else "common")
        A(f"  {name:<12}: {c:>6}  [{tier}]")
    A("\n[Training]")
    for k in ["epochs", "batch_size", "lr", "weight_decay", "optimizer",
              "scheduler", "warmup_ratio", "early_stopping_patience",
              "threshold", "loss", "use_weighted_sampler"]:
        A(f"  {k:<28}: {train_cfg.get(k, '?')}")
    A("")

    text = "\n".join(lines)
    print(text)
    out_path = os.path.join(run_dir, "logs", "config.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"[config] Saved → {out_path}\n")


# =============================================================================
#  CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, default="config/config.yaml")
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()
    train(config_path=args.config, run_dir=args.run_dir)
