from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from transformers import get_cosine_schedule_with_warmup
import yaml

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_run_dir(cfg: dict) -> Tuple[str, str]:
    base    = cfg.get("run_base_dir", ".")
    s1_name = cfg["stage1"]["model"]["name"]
    s2_name = cfg["stage2"]["model"]["name"]
    base_name = f"{s1_name}+{s2_name}"

    root = os.path.join(base, "run_2_stage")
    os.makedirs(root, exist_ok=True)

    candidate = os.path.join(root, base_name)
    if not os.path.exists(candidate):
        run_dir  = candidate
        run_name = base_name
    else:
        n = 2
        while True:
            name = f"{base_name}_{n}"
            candidate = os.path.join(root, name)
            if not os.path.exists(candidate):
                run_dir  = candidate
                run_name = name
                break
            n += 1

    for sub in [
        "checkpoints",
        "logs",
        os.path.join("results", "stage1"),
        os.path.join("results", "stage2"),
        os.path.join("results", "end2end"),
    ]:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    return run_dir, run_name


def get_existing_run_dir(cfg: dict) -> str:
    """
    Return the path of the LATEST existing run directory for this model pair.
    Used by test.py to find checkpoints without creating a new run.

    Raises FileNotFoundError if no run exists yet.
    """
    base    = cfg.get("run_base_dir", ".")
    s1_name = cfg["stage1"]["model"]["name"]
    s2_name = cfg["stage2"]["model"]["name"]
    base_name = f"{s1_name}+{s2_name}"
    root    = os.path.join(base, "run_2_stage")

    candidates = []
    base_path = os.path.join(root, base_name)
    if os.path.isdir(base_path):
        candidates.append((1, base_path))

    n = 2
    while True:
        p = os.path.join(root, f"{base_name}_{n}")
        if os.path.isdir(p):
            candidates.append((n, p))
            n += 1
        else:
            break

    if not candidates:
        raise FileNotFoundError(
            f"No run directory found for '{base_name}' under '{root}'. "
            f"Run train.py first."
        )

    # Return the latest (highest n)
    _, latest = sorted(candidates)[-1]
    return latest

def get_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    train_cfg     = cfg["training"]
    name          = train_cfg.get("optimizer", "adamw").lower()
    lr            = float(train_cfg.get("lr",           2e-5))
    wd            = float(train_cfg.get("weight_decay", 0.01))
    training_mode = train_cfg.get("training_mode", "finetune").lower().strip()

    def _make_optimizer(param_groups):
        if name == "adamw": return optim.AdamW(param_groups)
        if name == "adam":  return optim.Adam(param_groups)
        if name == "sgd":   return optim.SGD(param_groups, momentum=0.9)
        raise ValueError(f"Unknown optimizer: '{name}'")

    if training_mode == "freeze_backbone":
        head_params = [p for n, p in model.named_parameters()
                       if "classifiers" in n and p.requires_grad]
        if not head_params:
            raise RuntimeError("freeze_backbone: không tìm thấy tham số trainable nào!")
        param_groups = [{"params": head_params, "lr": lr * 3, "weight_decay": wd}]
        print(f"[optimizer] freeze_backbone — chỉ train head, lr={lr*3:.2e}")
        return _make_optimizer(param_groups)

    if training_mode == "from_scratch":
        all_params = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": all_params, "lr": lr, "weight_decay": wd}]
        print(f"[optimizer] from_scratch — train toàn bộ, lr={lr:.2e}")
        return _make_optimizer(param_groups)

    if training_mode == "llrd":
        decay_rate = float(train_cfg.get("llrd_decay_rate", 0.8))
        return _build_llrd_optimizer(model, lr, wd, decay_rate, name)

    backbone_params, classifier_params = [], []
    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (classifier_params if "classifiers" in pname else backbone_params).append(param)

    param_groups = [
        {"params": backbone_params,   "lr": lr,      "weight_decay": wd},
        {"params": classifier_params, "lr": lr * 3,  "weight_decay": wd},
    ]
    print(f"[optimizer] finetune — backbone lr={lr:.2e}, head lr={lr*3:.2e}")
    return _make_optimizer(param_groups)


def _build_llrd_optimizer(
    model:      nn.Module,
    base_lr:    float,
    wd:         float,
    decay_rate: float,
    opt_name:   str,
) -> optim.Optimizer:
    backbone = model.backbone
    encoder_layers = None
    for attr in ["encoder.layer", "encoder.layers", "transformer.layer"]:
        obj = backbone
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            encoder_layers = list(obj)
            break
        except AttributeError:
            continue

    if encoder_layers is None:
        print("[LLRD] Cảnh báo: không nhận diện được encoder layers, fallback sang finetune.")
        backbone_params    = [p for n, p in model.named_parameters()
                              if "classifiers" not in n and p.requires_grad]
        classifier_params  = [p for n, p in model.named_parameters()
                              if "classifiers" in n and p.requires_grad]
        param_groups = [
            {"params": backbone_params,  "lr": base_lr,     "weight_decay": wd},
            {"params": classifier_params,"lr": base_lr * 3, "weight_decay": wd},
        ]
        return optim.AdamW(param_groups)

    n_layers = len(encoder_layers)
    param_groups: list = []
    assigned: set = set()

    head_params = [(n, p) for n, p in model.named_parameters()
                   if "classifiers" in n and p.requires_grad]
    if head_params:
        param_groups.append({
            "params":       [p for _, p in head_params],
            "lr":           base_lr * 3,
            "weight_decay": wd,
            "name":         "head",
        })
        assigned.update(id(p) for _, p in head_params)

    for depth, layer in enumerate(reversed(encoder_layers)):
        layer_lr = base_lr * (decay_rate ** depth)
        layer_params = [(n, p) for n, p in layer.named_parameters(recurse=True)
                        if p.requires_grad and id(p) not in assigned]
        if layer_params:
            param_groups.append({
                "params":       [p for _, p in layer_params],
                "lr":           layer_lr,
                "weight_decay": wd,
                "name":         f"layer_{n_layers - 1 - depth}",
            })
            assigned.update(id(p) for _, p in layer_params)

    pooler_params = [(n, p) for n, p in backbone.named_parameters()
                     if "pooler" in n and p.requires_grad and id(p) not in assigned]
    if pooler_params:
        param_groups.append({
            "params":       [p for _, p in pooler_params],
            "lr":           base_lr,
            "weight_decay": wd,
            "name":         "pooler",
        })
        assigned.update(id(p) for _, p in pooler_params)

    emb_lr = base_lr * (decay_rate ** n_layers)
    emb_params = [(n, p) for n, p in backbone.named_parameters()
                  if p.requires_grad and id(p) not in assigned]
    if emb_params:
        param_groups.append({
            "params":       [p for _, p in emb_params],
            "lr":           emb_lr,
            "weight_decay": wd,
            "name":         "embeddings",
        })

    print(f"[optimizer] llrd — {n_layers} layers, decay={decay_rate}")
    print(f"  head      lr={base_lr*3:.2e}")
    for g in param_groups:
        if g.get("name", "").startswith("layer_"):
            top_name = f"layer_{n_layers-1}"
            bot_name = "layer_0"
            if g["name"] == top_name:
                print(f"  {g['name']} (top) lr={g['lr']:.2e}")
            elif g["name"] == bot_name:
                print(f"  {g['name']} (bottom) lr={g['lr']:.2e}")
    print(f"  embeddings lr={emb_lr:.2e}")

    if opt_name == "adamw": return optim.AdamW(param_groups)
    if opt_name == "adam":  return optim.Adam(param_groups)
    if opt_name == "sgd":   return optim.SGD(param_groups, momentum=0.9)
    raise ValueError(f"Unknown optimizer: '{opt_name}'")


def get_scheduler(optimizer, cfg: dict, num_training_steps: int):
    train_cfg    = cfg["training"]
    name         = train_cfg.get("scheduler", "cosine_warmup").lower()
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    num_warmup   = int(num_training_steps * warmup_ratio)

    if name == "cosine_warmup":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup,
            num_training_steps=num_training_steps,
        )
    if name == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(train_cfg.get("epochs", 10))
        )
    if name == "step":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=int(train_cfg.get("step_size", 3)),
            gamma=float(train_cfg.get("gamma", 0.1)),
        )
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: '{name}'")

class AverageMeter:
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

def apply_threshold(probs: np.ndarray, threshold) -> np.ndarray:
    """
    (N, C) probs → (N, C) int32 predictions.
    threshold: scalar float or np.ndarray (C,).
    """
    return (probs >= threshold).astype(np.int32)


def find_best_thresholds(
    probs:      np.ndarray,
    labels:     np.ndarray,
    candidates: np.ndarray = None,
    metric:     str = "f1",
) -> np.ndarray:
    """Per-class threshold search on val set. Returns (C,) array."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    if candidates is None:
        candidates = np.arange(0.05, 0.95, 0.05)

    score_fn = {
        "f1":        lambda y, p: f1_score(y, p,        zero_division=0),
        "precision": lambda y, p: precision_score(y, p, zero_division=0),
        "recall":    lambda y, p: recall_score(y, p,    zero_division=0),
    }[metric]

    num_classes     = probs.shape[1]
    best_thresholds = np.full(num_classes, 0.5)
    for c in range(num_classes):
        best_score, best_t = -1.0, 0.5
        for t in candidates:
            s = score_fn(labels[:, c], (probs[:, c] >= t).astype(int))
            if s > best_score:
                best_score, best_t = s, t
        best_thresholds[c] = best_t
    return best_thresholds

def save_config_summary(
    run_dir:      str,
    cfg:          dict,
    stage:        str,
    info:         dict,
    n_params:     int,
    n_train:      int,
    n_val:        int,
    n_test:       int,
) -> str:
    stage_cfg = cfg[stage]
    train_cfg = stage_cfg["training"]
    data_cfg  = cfg["data"]
    model_name = stage_cfg["model"]["name"]

    from src.dataloader import BACKBONE_REGISTRY, EMOTION_NAMES
    pretrained = BACKBONE_REGISTRY.get(model_name, {}).get("pretrained", "?")

    tier_indices  = info.get("tier_indices", {})
    label_counts  = info.get("label_counts", {})

    lines = []
    A = lines.append

    A("=" * 60)
    A(f"  Config Summary  —  {stage.upper()}")
    A(f"  Run : {os.path.basename(run_dir)}")
    A(f"  Path: {run_dir}")
    A("=" * 60)

    A("\n[Model]")
    A(f"  name        : {model_name}")
    A(f"  pretrained  : {pretrained}")
    A(f"  dropout     : {stage_cfg['model'].get('dropout', 0.1)}")
    A(f"  params      : {n_params:,}")
    A(f"  num_labels  : {info.get('num_labels', '?')}")

    A("\n[Data]")
    A(f"  train_file  : {data_cfg.get('train_file', '?')}")
    A(f"  auto_split  : {data_cfg.get('auto_split', True)}")
    A(f"  val_ratio   : {data_cfg.get('val_ratio', 0.1)}")
    A(f"  test_ratio  : {data_cfg.get('test_ratio', 0.1)}")
    A(f"  max_length  : {data_cfg.get('max_length', 128)}")
    A(f"  seed        : {data_cfg.get('seed', 42)}")
    A(f"  n_train     : {n_train}")
    A(f"  n_val       : {n_val}")
    A(f"  n_test      : {n_test}")

    if stage == "stage2":
        A("\n[Class Counts & Tiers]")
        import numpy as np
        counts = [label_counts.get(n, 0) for n in EMOTION_NAMES]
        median = float(np.median(counts)) if counts else 1.0
        vr_div = float(train_cfg.get("very_rare_divisor", 3.0))
        r_div  = float(train_cfg.get("rare_divisor",      1.0))
        for i, name in enumerate(EMOTION_NAMES):
            c = label_counts.get(name, 0)
            tier = ("very_rare" if i in tier_indices.get("very_rare", []) else
                    "rare"      if i in tier_indices.get("rare",      []) else "common")
            A(f"  {name:<12}: {c:>6}  [{tier}]")
        A(f"  median={median:.0f}  thresholds: very_rare<{median/vr_div:.0f}, rare<{median/r_div:.0f}")

    elif stage == "stage1":
        A("\n[Class Counts]")
        A(f"  has_emotion : {label_counts.get('has_emotion', '?')}")
        A(f"  neutral     : {label_counts.get('neutral', '?')}")
    A("\n[Augmentation]")
    aug = bool(train_cfg.get("augment_rare", False))
    A(f"  augment_rare: {aug}")
    if aug and stage == "stage2":
        A(f"  aug_copies_very_rare : {train_cfg.get('aug_copies_very_rare', 4)}")
        A(f"  aug_copies_rare      : {train_cfg.get('aug_copies_rare',      2)}")
        A(f"  aug_copies_common    : {train_cfg.get('aug_copies_common',    0)}")

    A("\n[Sampler]")
    use_s = bool(train_cfg.get("use_weighted_sampler", stage == "stage2"))
    A(f"  use_weighted_sampler: {use_s}")
    if use_s and stage == "stage2":
        A(f"  sampler_power       : {train_cfg.get('sampler_power',   2.0)}")
        A(f"  boost_very_rare     : {train_cfg.get('boost_very_rare', 5.0)}")
        A(f"  boost_rare          : {train_cfg.get('boost_rare',      3.0)}")
        A(f"  boost_common        : {train_cfg.get('boost_common',    1.0)}")

    A("\n[pos_weight]")
    if stage == "stage1":
        A(f"  pw_scale            : {train_cfg.get('pw_scale', 1.0)}")
    else:
        A(f"  pw_scale_very_rare  : {train_cfg.get('pw_scale_very_rare', 2.0)}")
        A(f"  pw_scale_rare       : {train_cfg.get('pw_scale_rare',      1.5)}")
        A(f"  pw_scale_common     : {train_cfg.get('pw_scale_common',    1.0)}")

    A("\n[Loss]")
    loss_name = train_cfg.get("loss", "bce_weighted")
    A(f"  loss                : {loss_name}")
    if loss_name in ("bce", "bce_weighted"):
        pass   
    elif loss_name == "focal_bce":
        A(f"  focal_gamma         : {train_cfg.get('focal_gamma', 2.0)}")
    elif loss_name == "asymmetric":
        A(f"  asl_gamma_pos       : {train_cfg.get('asl_gamma_pos', 0.0)}")
        A(f"  asl_gamma_neg       : {train_cfg.get('asl_gamma_neg', 4.0)}")
        A(f"  asl_clip            : {train_cfg.get('asl_clip',      0.05)}")
    elif loss_name == "per_class_asl":
        A(f"  very_rare  gamma_pos: {train_cfg.get('asl_gamma_pos_very_rare', 0.0)}")
        A(f"  very_rare  gamma_neg: {train_cfg.get('asl_gamma_neg_very_rare', 0.5)}")
        A(f"  very_rare  clip     : {train_cfg.get('asl_clip_very_rare',      0.0)}")
        A(f"  rare       gamma_pos: {train_cfg.get('asl_gamma_pos_rare',      0.5)}")
        A(f"  rare       gamma_neg: {train_cfg.get('asl_gamma_neg_rare',      1.0)}")
        A(f"  rare       clip     : {train_cfg.get('asl_clip_rare',           0.0)}")
        A(f"  common     gamma_pos: {train_cfg.get('asl_gamma_pos_common',    1.0)}")
        A(f"  common     gamma_neg: {train_cfg.get('asl_gamma_neg_common',    2.0)}")
        A(f"  common     clip     : {train_cfg.get('asl_clip_common',         0.0)}")

    A("\n[Training]")
    A(f"  epochs              : {train_cfg.get('epochs', '?')}")
    A(f"  batch_size          : {train_cfg.get('batch_size', 32)}")
    A(f"  lr                  : {train_cfg.get('lr', 2e-5)}")
    A(f"  weight_decay        : {train_cfg.get('weight_decay', 0.01)}")
    A(f"  optimizer           : {train_cfg.get('optimizer', 'adamw')}")
    A(f"  scheduler           : {train_cfg.get('scheduler', 'cosine_warmup')}")
    A(f"  warmup_ratio        : {train_cfg.get('warmup_ratio', 0.1)}")
    A(f"  early_stop_patience : {train_cfg.get('early_stopping_patience', 3)}")
    A(f"  threshold           : {train_cfg.get('threshold', 0.5)}")
    A("")

    text = "\n".join(lines)
    print(text)

    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    out_path = os.path.join(run_dir, "logs", f"config_{stage}.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"[config] Saved → {out_path}\n")
    return out_path


def find_best_threshold_binary(
    probs:      np.ndarray,
    labels:     np.ndarray,
    candidates: np.ndarray = None,
    metric:     str = "f1",
) -> float:
    """Scalar threshold search for Stage 1 (binary)."""
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    probs  = probs.flatten()
    labels = labels.flatten()
    if candidates is None:
        candidates = np.arange(0.05, 0.95, 0.05)

    score_fn = {
        "f1":        lambda y, p: f1_score(y, p,        zero_division=0),
        "accuracy":  lambda y, p: accuracy_score(y, p),
        "precision": lambda y, p: precision_score(y, p, zero_division=0),
        "recall":    lambda y, p: recall_score(y, p,    zero_division=0),
    }[metric]

    best_score, best_t = -1.0, 0.5
    for t in candidates:
        s = score_fn(labels, (probs >= t).astype(int))
        if s > best_score:
            best_score, best_t = s, t
    return best_t
