"""
src/test.py
Single-Stage End-to-End Ekman Classification — Evaluation.

Results saved under:
    <run_dir>/results/
        report.txt
        metrics.csv
        per_class.csv
        per_class_f1.png
        thresholds.png
        heatmap.png
        pr_curve.png
        confusion.png
        confusion_aggregate.png

CLI:
    python src/test.py
    python src/test.py --config config/config.yaml
    python src/test.py --run_dir /content/drive/MyDrive/run_e2e/deberta
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, hamming_loss,
    f1_score, precision_score, recall_score,
    precision_recall_curve, average_precision_score,
)
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataloader import get_dataloaders, CLASS_NAMES, NUM_CLASSES
from src.train import build_model, get_existing_run_dir
from src.utils import load_config, set_seed, apply_threshold, find_best_thresholds


# =============================================================================
#  Plotting helpers
# =============================================================================

def _plot_hbar_f1(f1s, names, path, title=""):
    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.8)))
    y       = np.arange(len(names))
    colors  = ["#e74c3c" if v < 0.5 else "#f39c12" if v < 0.7 else "#2ecc71" for v in f1s]
    bars    = ax.barh(y, f1s, color=colors, edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()
    ax.set_xlim(0, 1.1); ax.set_xlabel("F1 Score"); ax.set_title(title)
    ax.axvline(np.mean(f1s), color="black", linestyle="--", linewidth=1,
               label=f"Mean={np.mean(f1s):.3f}")
    ax.legend(fontsize=9)
    for bar, v in zip(bars, f1s):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8)
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_threshold_bar(ts, names, path, title=""):
    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.8)))
    y = np.arange(len(names))
    colors = ["#e74c3c" if t < 0.4 else "#2ecc71" if t > 0.6 else "#3498db" for t in ts]
    ax.barh(y, ts, color=colors, edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()
    ax.axvline(0.5, color="black", linestyle="--", label="default=0.5")
    ax.set_xlabel("Threshold"); ax.set_title(title); ax.set_xlim(0, 1.05)
    ax.legend(fontsize=8)
    for yi, v in zip(y, ts):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8)
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_heatmap(probs, labels, names, path, n=60, title=""):
    idx = np.random.default_rng(0).choice(len(probs), size=min(n, len(probs)), replace=False)
    fig, ax = plt.subplots(figsize=(max(12, len(names) * 1.5), 8))
    im = ax.imshow(probs[idx], aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    rows, cols = np.where(labels[idx] == 1)
    ax.scatter(cols, rows, marker="x", color="blue", s=20, linewidths=1, label="True")
    ax.legend(fontsize=8)
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_confusion_aggregate(cm, names, path, title=""):
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(max(6, len(names)), max(5, len(names) - 1)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_pr_curve(y_true, y_probs, names, path, title=""):
    palette = plt.cm.get_cmap("tab10", len(names))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(names):
        prec, rec, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        ap = average_precision_score(y_true[:, i], y_probs[:, i])
        ax.plot(rec, prec, color=palette(i), linewidth=1.8, label=f"{name}  AP={ap:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05); ax.set_title(title)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_confusion_multilabel(y_true, y_pred, names, path, title=""):
    n = len(names)
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes = np.array(axes).flatten()
    for i, name in enumerate(names):
        cm   = confusion_matrix(y_true[:, i], y_pred[:, i])
        norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax   = axes[i]
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        fontsize=9, color="white" if norm[r, c] > 0.5 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=7)
        ax.set_yticklabels(["True 0", "True 1"], fontsize=7)
        ax.set_title(name, fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# =============================================================================
#  Inference helper
# =============================================================================

def _infer(model, loader, device, desc="Inference") -> Tuple[np.ndarray, np.ndarray]:
    probs_list, labels_list = [], []
    pbar = tqdm(loader, desc=desc, leave=True, dynamic_ncols=True,
                unit="batch", mininterval=1.0)
    with torch.no_grad():
        for batch in pbar:
            logits = model(
                batch["input_ids"].to(device,      non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
            )
            probs_list.append(torch.sigmoid(logits).float().cpu().numpy())
            labels_list.append(batch["labels"].float().cpu().numpy())
    pbar.close()
    return np.vstack(probs_list), np.vstack(labels_list)


def _load_checkpoint(run_dir: str, cfg: dict, device: torch.device):
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{ckpt_path}'\nRun: python src/train.py first."
        )
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg["e2e"], num_labels=ckpt["num_labels"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[test] Loaded checkpoint  epoch={ckpt.get('epoch','?')}  "
          f"val_macro_f1={ckpt.get('val_macro_f1', float('nan')):.4f}")
    return model, ckpt


# =============================================================================
#  Evaluation
# =============================================================================

def evaluate(
    config_path: str = "config/config.yaml",
    run_dir:     str = None,
) -> Dict:
    cfg    = load_config(config_path)
    set_seed(cfg["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if run_dir is None:
        run_dir = get_existing_run_dir(cfg)

    out_dir = os.path.join(run_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*62}\n  Evaluation — 7 Classes\n{'='*62}")

    model, ckpt = _load_checkpoint(run_dir, cfg, device)
    _, val_loader, test_loader, _ = get_dataloaders(cfg)

    print("\n[test] Per-class threshold search on val set...")
    vp, vl  = _infer(model, val_loader, device, "Val")
    best_ts = find_best_thresholds(vp, vl, candidates=np.arange(0.05, 0.95, 0.05), metric="f1")
    for i, name in enumerate(CLASS_NAMES):
        vf1 = f1_score(vl[:, i], (vp[:, i] >= best_ts[i]).astype(int), zero_division=0)
        print(f"  {name:<12}: t={best_ts[i]:.2f}  val_F1={vf1:.4f}")

    tp, tl = _infer(model, test_loader, device, "Test")
    preds  = apply_threshold(tp, best_ts)

    micro_f1    = f1_score(tl, preds, average="micro",    zero_division=0)
    macro_f1    = f1_score(tl, preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(tl, preds, average="weighted", zero_division=0)
    hamming     = hamming_loss(tl, preds)
    subset_acc  = float((preds == tl).all(axis=1).mean())
    per_f1      = f1_score(tl, preds, average=None, zero_division=0)
    per_p       = precision_score(tl, preds, average=None, zero_division=0)
    per_r       = recall_score(tl, preds, average=None, zero_division=0)
    per_sup     = tl.sum(axis=0).astype(int)
    report      = classification_report(tl, preds, target_names=CLASS_NAMES, zero_division=0)

    print(f"\n{'='*62}")
    print(f"  Micro  F1    : {micro_f1:.4f}")
    print(f"  Macro  F1    : {macro_f1:.4f}")
    print(f"  Weighted F1  : {weighted_f1:.4f}")
    print(f"  Hamming Loss : {hamming:.4f}")
    print(f"  Subset Acc   : {subset_acc:.4f}")
    print(f"{'='*62}\n{report}")

    model_name = cfg["e2e"]["model"]["name"]

    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write("E2E Single-Stage — 7 Classes\n")
        f.write(f"Model: {model_name} | Epoch: {ckpt.get('epoch','?')}\n")
        f.write("Per-class thresholds:\n")
        for name, t in zip(CLASS_NAMES, best_ts):
            f.write(f"  {name:<12}: {t:.2f}\n")
        f.write(f"\nMicro F1: {micro_f1:.4f}\nMacro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\nHamming: {hamming:.4f}\n")
        f.write(f"Subset Acc: {subset_acc:.4f}\n\n{report}")

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "micro_f1", "macro_f1", "weighted_f1", "hamming", "subset_acc", "epoch"
        ])
        w.writeheader()
        w.writerow({"model": model_name, "micro_f1": round(micro_f1, 4),
                    "macro_f1": round(macro_f1, 4), "weighted_f1": round(weighted_f1, 4),
                    "hamming": round(hamming, 4), "subset_acc": round(subset_acc, 4),
                    "epoch": ckpt.get("epoch", "?")})

    with open(os.path.join(out_dir, "per_class.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "threshold", "precision", "recall", "f1", "support"])
        w.writeheader()
        for i, name in enumerate(CLASS_NAMES):
            w.writerow({"class": name, "threshold": round(float(best_ts[i]), 2),
                        "precision": round(float(per_p[i]), 4),
                        "recall":    round(float(per_r[i]), 4),
                        "f1":        round(float(per_f1[i]), 4),
                        "support":   int(per_sup[i])})

    _plot_hbar_f1(per_f1, CLASS_NAMES,
                  os.path.join(out_dir, "per_class_f1.png"),
                  "Per-Class F1 (7 classes, full test set)")
    _plot_threshold_bar(best_ts, CLASS_NAMES,
                        os.path.join(out_dir, "thresholds.png"),
                        "Per-Class Optimal Threshold")
    _plot_heatmap(tp, tl, CLASS_NAMES,
                  os.path.join(out_dir, "heatmap.png"),
                  title="Predicted Probabilities (test subset)")
    _plot_pr_curve(tl, tp, CLASS_NAMES,
                   os.path.join(out_dir, "pr_curve.png"),
                   "Per-Class PR Curves (7 classes, full test set)")
    _plot_confusion_multilabel(tl, preds, CLASS_NAMES,
                               os.path.join(out_dir, "confusion.png"),
                               "Per-Class Confusion Matrices (test set)")

    single_mask = tl.sum(axis=1) == 1
    if single_mask.sum() > 0:
        cm7 = confusion_matrix(
            np.argmax(tl[single_mask], axis=1),
            np.argmax(preds[single_mask], axis=1),
            labels=list(range(NUM_CLASSES))
        )
        _plot_confusion_aggregate(cm7, CLASS_NAMES,
                                  os.path.join(out_dir, "confusion_aggregate.png"),
                                  "Aggregated Confusion Matrix (single-label samples)")

    print(f"[test] Saved → {out_dir}")

    return {
        "micro_f1": micro_f1, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "hamming": hamming, "subset_accuracy": subset_acc,
        "best_thresholds": best_ts, "out_dir": out_dir,
    }


# =============================================================================
#  CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, default="config/config.yaml")
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()
    evaluate(config_path=args.config, run_dir=args.run_dir)
