"""
src/train.py
Single-Stage End-to-End Ekman Classification — Training.

Task   : 7-class multi-label (6 emotions + neutral), ALL samples.
Model  : EncoderForClassification — pretrained Transformer + 7 independent heads.

Output structure (YOLO-style auto-increment):
    <run_base_dir>/run_e2e/<model_name>[_N]/
        checkpoints/best.pth
        logs/training_log.csv
        logs/config.txt
        results/               ← filled by test.py

CLI:
    python src/train.py
    python src/train.py --config config/config.yaml
    python src/train.py --run_dir /path/to/existing/run
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
    sentence_predict,
    CLASS_NAMES, NUM_CLASSES,
    BACKBONE_REGISTRY,
)
from src.utils import (
    AverageMeter, get_optimizer, get_scheduler,
    load_config, set_seed, apply_threshold,
)
from models.loss import get_loss_fn


#  Model

class EncoderForClassification(nn.Module):
    """
    Pretrained Transformer encoder with 7 independent binary classification heads
    (6 Ekman emotions + neutral).

    Architecture:
        [CLS] token representation → Dropout → 7 × Linear(hidden, 1) → (B, 7) logits

    Each head is trained independently; sigmoid is applied at inference time.
    """

    def __init__(
        self,
        pretrained_name: str,
        num_labels:      int   = NUM_CLASSES,
        dropout:         float = 0.1,
        from_scratch:    bool  = False,
    ) -> None:
        super().__init__()
        if from_scratch:
            from transformers import AutoConfig
            config        = AutoConfig.from_pretrained(pretrained_name)
            self.backbone = AutoModel.from_config(config)
        else:
            self.backbone = AutoModel.from_pretrained(pretrained_name)

        self.dropout     = nn.Dropout(dropout)
        hidden           = self.backbone.config.hidden_size
        self.classifiers = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(num_labels)]
        )
        self.num_labels  = num_labels

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, L) long tensor
            attention_mask: (B, L) long tensor
        Returns:
            logits: (B, num_labels) float tensor
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return torch.cat([h(cls) for h in self.classifiers], dim=1)


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone parameters; only classification heads remain trainable."""
    for name, param in model.named_parameters():
        if "classifiers" not in name:
            param.requires_grad = False


def build_model(stage_cfg: dict, num_labels: int = NUM_CLASSES) -> nn.Module:
    """Build and return the EncoderForClassification model."""
    name = stage_cfg["model"]["name"].lower()
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {' | '.join(BACKBONE_REGISTRY)}"
        )
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
        _freeze_backbone(model)
        frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[build_model] freeze_backbone — frozen={frozen:,}  trainable={trainable:,}")
    elif training_mode == "from_scratch":
        print("[build_model] from_scratch — backbone initialised with random weights")
    elif training_mode == "llrd":
        print("[build_model] llrd — Layerwise LR Decay applied by optimizer")
    else:
        print("[build_model] finetune — full end-to-end fine-tuning")

    return model


#  Run directory  (YOLO-style auto-increment)
def get_run_dir(cfg: dict) -> Tuple[str, str]:
    """Return (run_dir, run_name), creating subdirectories as needed."""
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
    """Return the latest existing run directory for this model."""
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


#  Training loop helpers
def _train_epoch(
    model:        nn.Module,
    loader:       torch.utils.data.DataLoader,
    criterion:    nn.Module,
    optimizer:    torch.optim.Optimizer,
    scheduler,
    scaler:       GradScaler | None,
    device:       torch.device,
    epoch:        int,
    total_epochs: int,
    use_amp:      bool       = False,
    amp_dtype:    torch.dtype = torch.float16,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    loss_meter = AverageMeter("loss")
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [train]",
        leave=True, dynamic_ncols=True, unit="batch", mininterval=1.0,
    )
    for batch in pbar:
        ids  = batch["input_ids"].to(device,      non_blocking=True)
        mask = batch["attention_mask"].to(device,  non_blocking=True)
        lbls = batch["labels"].to(device,          non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(ids, mask)
            loss   = criterion(logits, lbls)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"\n[WARNING] NaN/Inf gradient — skipping batch, resetting scaler")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
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
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    pbar.close()
    return loss_meter.avg


def _val_epoch(
    model:      nn.Module,
    val_texts:  List[str],
    val_labels: np.ndarray,
    tokenizer,
    max_length: int,
    batch_size: int,
    criterion:  nn.Module,
    device:     torch.device,
    threshold:  float       = 0.5,
    use_amp:    bool        = False,
    amp_dtype:  torch.dtype = torch.float16,
    window:     int         = 0,
) -> Tuple[float, Dict[str, float]]:
    """
    Validate using sentence-level inference with context window.

    F1 metrics are computed from sentence-split predictions (max aggregation).
    Val loss is approximated from a single truncated forward pass per sample
    (used for monitoring convergence only, not for early stopping).
    """
    # Sentence-split predictions for F1
    probs, labels = sentence_predict(
        model, val_texts, val_labels,
        tokenizer, max_length, batch_size, device, window=window,
    )
    preds = apply_threshold(probs, threshold)
    metrics = {
        "micro_f1":    float(f1_score(labels, preds, average="micro",    zero_division=0)),
        "macro_f1":    float(f1_score(labels, preds, average="macro",    zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }

    # Approximate val loss (single truncated pass — for monitoring only)
    model.eval()
    loss_meter = AverageMeter("val_loss")
    from src.dataloader import EkmanDataset
    val_ds     = EkmanDataset(val_texts, val_labels, tokenizer, max_length)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    with torch.no_grad():
        for batch in val_loader:
            ids  = batch["input_ids"].to(device,      non_blocking=True)
            mask = batch["attention_mask"].to(device,  non_blocking=True)
            lbls = batch["labels"].to(device,          non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(ids, mask)
                loss   = criterion(logits, lbls)
            loss_meter.update(loss.item(), lbls.size(0))

    return loss_meter.avg, metrics


#  Main training function

def train(
    config_path: str = "config/config.yaml",
    run_dir:     str = None,
) -> Dict:
    """
    Train the single-stage 7-class model.

    Returns:
        dict with keys: run_dir, run_name, best_val_macro_f1,
                        best_epoch, best_metrics, log_path, checkpoint_path.
    """
    cfg       = load_config(config_path)
    stage_cfg = cfg["e2e"]
    train_cfg = stage_cfg["training"]
    set_seed(cfg["data"]["seed"])

    epochs    = int(train_cfg["epochs"])
    patience  = int(train_cfg.get("early_stopping_patience", 5))
    threshold = float(train_cfg.get("threshold",             0.5))
    window    = int(train_cfg.get("inference_window",        0))

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_key = stage_cfg["model"]["name"].lower()

    # AMP setup
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

    # Run directory
    if run_dir is None:
        run_dir, run_name = get_run_dir(cfg)
        print(f"\n{'='*62}\n  New run  : {run_name}\n  Path     : {run_dir}\n{'='*62}")
    else:
        run_name = os.path.basename(run_dir)
        print(f"\n{'='*62}\n  Resuming : {run_name}\n{'='*62}")

    print(f"[train] Device={device}  AMP={use_amp} ({amp_dtype_str})")

    # Data
    train_loader, _, _, info = get_dataloaders(cfg)
    pos_weight   = info["pos_weight"]
    num_labels   = info["num_labels"]
    pretrained   = info["pretrained"]
    tokenizer    = info["tokenizer"]
    max_length   = info["max_length"]
    batch_size   = info["batch_size"]
    val_texts    = info["val_texts"]
    val_labels_7 = info["val_labels_7"]

    # Model
    model      = build_model(stage_cfg, num_labels=num_labels).to(device)
    model_name = stage_cfg["model"]["name"]
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model  : {model_name}  ({pretrained})")
    print(f"[train] Params : {n_params:,}  |  Labels: {num_labels}")

    # Loss / Optimizer / Scheduler
    cfg_for_loss = {"training": dict(train_cfg)}
    criterion    = get_loss_fn(cfg_for_loss, device, pos_weight=pos_weight)
    optimizer    = get_optimizer(model, cfg_for_loss)
    total_steps  = len(train_loader) * epochs
    scheduler    = get_scheduler(optimizer, cfg_for_loss, num_training_steps=total_steps)
    scaler       = GradScaler() if use_scaler else None

    # Logging
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pth")
    log_path  = os.path.join(run_dir, "logs",        "training_log.csv")
    csv_fields = [
        "epoch", "train_loss", "val_loss",
        "val_micro_f1", "val_macro_f1", "val_weighted_f1", "lr",
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    _save_config_summary(run_dir, cfg, info, n_params,
                         len(train_loader.dataset), len(val_texts))

    # Training loop
    best_val_macro_f1 = -1.0
    best_metrics:     Dict[str, float] = {}
    best_epoch        = 0
    no_improve        = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = _train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, epochs, use_amp=use_amp, amp_dtype=amp_dtype,
        )

        val_loss, val_metrics = _val_epoch(
            model, val_texts, val_labels_7,
            tokenizer, max_length, batch_size,
            criterion, device,
            threshold=threshold, use_amp=use_amp, amp_dtype=amp_dtype, window=window,
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

        row = {
            "epoch":           epoch,
            "train_loss":      round(train_loss, 6),
            "val_loss":        round(val_loss, 6),
            "val_micro_f1":    round(val_metrics.get("micro_f1",    0), 4),
            "val_macro_f1":    round(val_metrics.get("macro_f1",    0), 4),
            "val_weighted_f1": round(val_metrics.get("weighted_f1", 0), 4),
            "lr":              lr,
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        # Checkpoint on best val macro-F1
        if val_metrics.get("macro_f1", 0) > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_metrics      = val_metrics.copy()
            best_epoch        = epoch
            no_improve        = 0
            torch.save({
                "epoch":           epoch,
                "model_name":      model_name,
                "pretrained_name": pretrained,
                "num_labels":      num_labels,
                "model_state":     model.state_dict(),
                "val_macro_f1":    best_val_macro_f1,
                "val_metrics":     best_metrics,
                "threshold":       threshold,
            }, ckpt_path)
            print(f"  ✓ Checkpoint saved (epoch {epoch}  macro_f1={best_val_macro_f1:.4f})")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[train] Early stopping at epoch {epoch} "
                  f"({patience} consecutive epochs without improvement).")
            break

    print(f"\n[train] Done — best val_macro_f1={best_val_macro_f1:.4f} @ epoch {best_epoch}")
    for k, v in best_metrics.items():
        print(f"        {k}={v:.4f}")
    print(f"[train] Log  → {log_path}")

    return {
        "run_dir":           run_dir,
        "run_name":          run_name,
        "best_val_macro_f1": best_val_macro_f1,
        "best_epoch":        best_epoch,
        "best_metrics":      best_metrics,
        "log_path":          log_path,
        "checkpoint_path":   ckpt_path,
    }


#  Config summary

def _save_config_summary(
    run_dir:  str,
    cfg:      dict,
    info:     dict,
    n_params: int,
    n_train:  int,
    n_val:    int,
) -> None:
    stage_cfg    = cfg["e2e"]
    train_cfg    = stage_cfg["training"]
    data_cfg     = cfg["data"]
    model_name   = stage_cfg["model"]["name"]
    pretrained   = BACKBONE_REGISTRY.get(model_name, {}).get("pretrained", "?")
    label_counts = info.get("label_counts", {})

    lines = [
        "=" * 60,
        "  Config Summary — E2E Single-Stage (7-class)",
        f"  Run : {os.path.basename(run_dir)}",
        f"  Path: {run_dir}",
        "=" * 60,
        "\n[Model]",
        f"  name          : {model_name}",
        f"  pretrained    : {pretrained}",
        f"  dropout       : {stage_cfg['model'].get('dropout', 0.1)}",
        f"  params        : {n_params:,}",
        f"  num_labels    : {info.get('num_labels', NUM_CLASSES)}",
        f"  training_mode : {train_cfg.get('training_mode', 'finetune')}",
        "\n[Data]",
        f"  train_file    : {data_cfg.get('train_file', '?')}",
        f"  auto_split    : {data_cfg.get('auto_split', False)}",
        f"  max_length    : {data_cfg.get('max_length', 128)}",
        f"  num_workers   : {data_cfg.get('num_workers', 2)}",
        f"  n_train       : {n_train}  (post-augmentation)",
        f"  n_val         : {n_val}",
        "\n[Class Counts (post-augmentation)]",
    ]
    for name in CLASS_NAMES:
        lines.append(f"  {name:<12}: {label_counts.get(name, 0):>6}")
    lines += [
        "\n[Augmentation]",
        f"  enabled       : {train_cfg.get('augment', True)}",
        f"  method        : nlpaug SynonymAug (WordNet, aug_p=0.15)",
        f"  strategy      : classes below median oversampled to median count",
        "\n[Inference]",
        f"  method        : sentence splitting (NLTK) + context window",
        f"  inference_window : {train_cfg.get('inference_window', 0)} sentences each side",
        "\n[Training]",
    ]
    for k in ["epochs", "batch_size", "lr", "weight_decay", "optimizer",
              "scheduler", "warmup_ratio", "early_stopping_patience",
              "threshold", "loss", "asl_gamma_pos", "asl_gamma_neg", "asl_clip"]:
        lines.append(f"  {k:<28}: {train_cfg.get(k, '?')}")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    out_path = os.path.join(run_dir, "logs", "config.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"[config] Saved → {out_path}\n")


#  CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, default="config/config.yaml")
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()
    train(config_path=args.config, run_dir=args.run_dir)
