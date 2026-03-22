"""
src/test.py
Two-Stage Ekman Emotion Classification — Evaluation.

Three evaluation modes:
    evaluate_stage1()      Binary test (neutral vs emotion)
    evaluate_stage2()      Multi-label 6-emotion test (emotion-only samples)
    evaluate_end_to_end()  Full 7-class pipeline evaluation

Results saved under the run directory:
    <run_dir>/results/stage1/
    <run_dir>/results/stage2/
    <run_dir>/results/end2end/

CLI:
    python src/test.py --mode all
    python src/test.py --mode stage1
    python src/test.py --mode end2end
    python src/test.py --run_dir /content/drive/MyDrive/run_2_stage/electra+deberta --mode all
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
    f1_score, precision_score, recall_score, accuracy_score,
    precision_recall_curve, average_precision_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataloader import (
    get_dataloaders, get_raw_splits,
    EMOTION_NAMES, NUM_EMOTIONS, ALL_CLASS_NAMES,
    BACKBONE_REGISTRY, EkmanDataset,
)
from src.utils import (
    load_config, set_seed, apply_threshold,
    find_best_thresholds, find_best_threshold_binary,
    get_existing_run_dir,
)
from src.train import build_model


# =============================================================================
#  Plotting helpers
# =============================================================================

def _plot_f1_bar(f1s, names, path, title=""):
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.1), 5))
    colors  = ["#e74c3c" if v < 0.5 else "#f39c12" if v < 0.7 else "#2ecc71" for v in f1s]
    bars    = ax.bar(names, f1s, color=colors, edgecolor="white")
    ax.axhline(np.mean(f1s), color="black", linestyle="--", linewidth=1,
               label=f"Mean={np.mean(f1s):.3f}")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.legend()
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_hbar_f1(f1s, names, path, title=""):
    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.8)))
    y       = np.arange(len(names))
    colors  = ["#e74c3c" if v < 0.5 else "#f39c12" if v < 0.7 else "#2ecc71" for v in f1s]
    bars    = ax.barh(y, f1s, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1 Score")
    ax.set_title(title)
    ax.axvline(np.mean(f1s), color="black", linestyle="--", linewidth=1,
               label=f"Mean={np.mean(f1s):.3f}")
    ax.legend(fontsize=9)
    for bar, v in zip(bars, f1s):
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion(cm, names, path, title=""):
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(max(6, len(names)), max(5, len(names) - 1)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_threshold_bar(ts, names, path, title=""):
    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.8)))
    y = np.arange(len(names))
    colors = ["#e74c3c" if t < 0.4 else "#2ecc71" if t > 0.6 else "#3498db" for t in ts]
    ax.barh(y, ts, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.5, color="black", linestyle="--", label="default=0.5")
    ax.set_xlabel("Threshold")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=8)
    for yi, v in zip(y, ts):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pr_curve_binary(y_true, y_probs, path, title=""):
    """PR curve for binary classification (Stage 1)."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#2980b9", linewidth=2, label=f"AP = {ap:.3f}")
    ax.fill_between(recall, precision, alpha=0.15, color="#2980b9")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pr_curve_multiclass(y_true, y_probs, names, path, title=""):
    """Per-class PR curves for multi-label / multi-class (Stage 2 & E2E)."""
    n = y_true.shape[1]
    palette = plt.cm.get_cmap("tab10", n)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(names):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        ap = average_precision_score(y_true[:, i], y_probs[:, i])
        ax.plot(recall, precision, color=palette(i), linewidth=1.8,
                label=f"{name}  AP={ap:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion_multilabel(y_true, y_pred, names, path, title=""):
    """Per-class confusion matrix (2×2) tiled in a grid for multi-label."""
    n = len(names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes = np.array(axes).flatten()
    for i, name in enumerate(names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax = axes[i]
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
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
#  Inference helper
# =============================================================================

def _infer(model, loader, device, desc="Inference") -> Tuple[np.ndarray, np.ndarray]:
    probs_list, labels_list = [], []
    pbar = tqdm(loader, desc=desc, leave=True, dynamic_ncols=True, unit="batch")
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


def _load_checkpoint(run_dir: str, stage: str, cfg: dict, device: torch.device):
    ckpt_path = os.path.join(run_dir, "checkpoints", f"{stage}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{ckpt_path}'\n"
            f"Run: python src/train.py --stage {stage}"
        )
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg[stage], stage=stage, num_labels=ckpt["num_labels"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[test] Loaded {stage} checkpoint  epoch={ckpt.get('epoch','?')}  ")
         
    return model, ckpt


# =============================================================================
#  Evaluate Stage 1
# =============================================================================

def evaluate_stage1(
    config_path: str = "config/config.yaml",
    run_dir:     str = None,
) -> Dict:
    cfg    = load_config(config_path)
    set_seed(cfg["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if run_dir is None:
        run_dir = get_existing_run_dir(cfg)

    out_dir = os.path.join(run_dir, "results", "stage1")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*62}\n  Stage 1 Evaluation — Neutral vs Emotion\n{'='*62}")

    _, val_loader, test_loader, _ = get_dataloaders(cfg, stage="stage1")
    model, ckpt = _load_checkpoint(run_dir, "stage1", cfg, device)

    # Val threshold search
    print("\n[test] Searching best threshold on val set...")
    vp, vl = _infer(model, val_loader, device, "S1 Val")
    best_t = find_best_threshold_binary(
        vp.flatten(), vl.flatten(), candidates=np.arange(0.05, 0.95, 0.05), metric="f1"
    )
    vf1 = f1_score(vl.flatten(), (vp.flatten() >= best_t).astype(int), zero_division=0)
    print(f"[test] Best threshold: {best_t:.2f}  val_F1={vf1:.4f}")

    # Test inference
    tp, tl = _infer(model, test_loader, device, "S1 Test")
    y_true = tl.flatten().astype(int)
    y_pred = (tp.flatten() >= best_t).astype(int)

    acc     = accuracy_score(y_true, y_pred)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report   = classification_report(y_true, y_pred,
                                     target_names=["neutral", "has_emotion"], zero_division=0)

    print(f"\n{'='*62}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1 (emotion): {f1:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Recall      : {rec:.4f}")
    print(f"  Macro F1    : {macro_f1:.4f}")
    print(f"{'='*62}\n{report}")

    # Save
    model_name = cfg["stage1"]["model"]["name"]

    with open(os.path.join(out_dir, "stage1_report.txt"), "w") as f:
        f.write(f"Stage 1 — Binary (Neutral vs Has-Emotion)\n")
        f.write(f"Model: {model_name} | Epoch: {ckpt.get('epoch','?')}\n")
        f.write(f"Best threshold: {best_t:.2f}  (val F1={vf1:.4f})\n\n")
        f.write(f"Accuracy    : {acc:.4f}\nF1 (emotion): {f1:.4f}\n")
        f.write(f"Precision   : {prec:.4f}\nRecall      : {rec:.4f}\n")
        f.write(f"Macro F1    : {macro_f1:.4f}\n\n{report}")

    with open(os.path.join(out_dir, "stage1_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model","threshold","accuracy","f1_emotion","precision","recall","macro_f1","epoch"
        ])
        w.writeheader()
        w.writerow({"model": model_name, "threshold": round(best_t, 2),
                    "accuracy": round(acc, 4), "f1_emotion": round(f1, 4),
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "macro_f1": round(macro_f1, 4), "epoch": ckpt.get("epoch","?")})

    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion(cm, ["neutral", "has_emotion"],
                    os.path.join(out_dir, "stage1_confusion.png"),
                    "Stage 1 Confusion Matrix (test set)")
    _plot_pr_curve_binary(y_true, tp.flatten(),
                          os.path.join(out_dir, "stage1_pr_curve.png"),
                          "Stage 1 PR Curve — has_emotion (test set)")
    _plot_f1_bar(
        [f1_score(y_true, y_pred, pos_label=0, zero_division=0), f1],
        ["neutral", "has_emotion"],
        os.path.join(out_dir, "stage1_f1_bar.png"),
        "Stage 1 F1 Score (test set)",
    )
    print(f"[test] Saved → {out_dir}")

    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec,
            "macro_f1": macro_f1, "best_threshold": best_t, "out_dir": out_dir}


# =============================================================================
#  Evaluate Stage 2
# =============================================================================

def evaluate_stage2(
    config_path: str = "config/config.yaml",
    run_dir:     str = None,
) -> Dict:
    cfg    = load_config(config_path)
    set_seed(cfg["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if run_dir is None:
        run_dir = get_existing_run_dir(cfg)

    out_dir = os.path.join(run_dir, "results", "stage2")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*62}\n  Stage 2 Evaluation — 6 Ekman Emotions\n{'='*62}")

    _, val_loader, test_loader, _ = get_dataloaders(cfg, stage="stage2")
    model, ckpt = _load_checkpoint(run_dir, "stage2", cfg, device)

    # Per-class threshold search
    print("\n[test] Per-class threshold search on val set...")
    vp, vl = _infer(model, val_loader, device, "S2 Val")
    best_ts = find_best_thresholds(vp, vl, candidates=np.arange(0.05, 0.95, 0.05), metric="f1")
    for i, name in enumerate(EMOTION_NAMES):
        vf1 = f1_score(vl[:, i], (vp[:, i] >= best_ts[i]).astype(int), zero_division=0)
        print(f"  {name:<12}: t={best_ts[i]:.2f}  val_F1={vf1:.4f}")

    # Test
    tp, tl = _infer(model, test_loader, device, "S2 Test")
    preds   = apply_threshold(tp, best_ts)

    micro_f1    = f1_score(tl, preds, average="micro",    zero_division=0)
    macro_f1    = f1_score(tl, preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(tl, preds, average="weighted", zero_division=0)
    hamming     = hamming_loss(tl, preds)
    subset_acc  = float((preds == tl).all(axis=1).mean())
    per_f1      = f1_score(tl, preds, average=None, zero_division=0)
    per_p       = precision_score(tl, preds, average=None, zero_division=0)
    per_r       = recall_score(tl, preds, average=None, zero_division=0)
    per_sup     = tl.sum(axis=0).astype(int)
    report      = classification_report(tl, preds, target_names=EMOTION_NAMES, zero_division=0)

    print(f"\n{'='*62}")
    print(f"  Micro  F1    : {micro_f1:.4f}")
    print(f"  Macro  F1    : {macro_f1:.4f}")
    print(f"  Weighted F1  : {weighted_f1:.4f}")
    print(f"  Hamming Loss : {hamming:.4f}")
    print(f"  Subset Acc   : {subset_acc:.4f}")
    print(f"{'='*62}\n{report}")

    model_name = cfg["stage2"]["model"]["name"]

    with open(os.path.join(out_dir, "stage2_report.txt"), "w") as f:
        f.write(f"Stage 2 — Multi-Label 6 Ekman Emotions (emotion-only test)\n")
        f.write(f"Model: {model_name} | Epoch: {ckpt.get('epoch','?')}\n")
        f.write("Per-class thresholds:\n")
        for name, t in zip(EMOTION_NAMES, best_ts):
            f.write(f"  {name:<12}: {t:.2f}\n")
        f.write(f"\nMicro F1: {micro_f1:.4f}\nMacro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\nHamming: {hamming:.4f}\n")
        f.write(f"Subset Acc: {subset_acc:.4f}\n\n{report}")

    with open(os.path.join(out_dir, "stage2_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model","micro_f1","macro_f1","weighted_f1","hamming","subset_acc","epoch"
        ])
        w.writeheader()
        w.writerow({"model": model_name, "micro_f1": round(micro_f1, 4),
                    "macro_f1": round(macro_f1, 4), "weighted_f1": round(weighted_f1, 4),
                    "hamming": round(hamming, 4), "subset_acc": round(subset_acc, 4),
                    "epoch": ckpt.get("epoch", "?")})

    with open(os.path.join(out_dir, "stage2_per_class.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["emotion","threshold","precision","recall","f1","support"])
        w.writeheader()
        for i, name in enumerate(EMOTION_NAMES):
            w.writerow({"emotion": name, "threshold": round(float(best_ts[i]), 2),
                        "precision": round(float(per_p[i]), 4),
                        "recall": round(float(per_r[i]), 4),
                        "f1": round(float(per_f1[i]), 4),
                        "support": int(per_sup[i])})

    _plot_hbar_f1(per_f1, EMOTION_NAMES,
                  os.path.join(out_dir, "stage2_per_class_f1.png"),
                  "Stage 2 Per-Class F1 (emotion-only test set)")
    _plot_threshold_bar(best_ts, EMOTION_NAMES,
                        os.path.join(out_dir, "stage2_thresholds.png"),
                        "Stage 2 Per-Class Optimal Threshold")
    _plot_heatmap(tp, tl, EMOTION_NAMES,
                  os.path.join(out_dir, "stage2_heatmap.png"),
                  title="Stage 2 Predicted Probabilities (test subset)")
    _plot_pr_curve_multiclass(tl, tp, EMOTION_NAMES,
                              os.path.join(out_dir, "stage2_pr_curve.png"),
                              "Stage 2 Per-Class PR Curves (emotion-only test set)")
    _plot_confusion_multilabel(tl, preds, EMOTION_NAMES,
                               os.path.join(out_dir, "stage2_confusion.png"),
                               "Stage 2 Per-Class Confusion Matrices (test set)")
    print(f"[test] Saved → {out_dir}")

    return {"micro_f1": micro_f1, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "hamming": hamming, "subset_accuracy": subset_acc,
            "best_thresholds": best_ts, "out_dir": out_dir}


# =============================================================================
#  End-to-End Evaluation (7 classes)
# =============================================================================

def evaluate_end_to_end(
    config_path: str = "config/config.yaml",
    run_dir:     str = None,
) -> Dict:
    cfg    = load_config(config_path)
    set_seed(cfg["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if run_dir is None:
        run_dir = get_existing_run_dir(cfg)

    out_dir = os.path.join(run_dir, "results", "end2end")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*62}\n  End-to-End Evaluation — 7 Classes\n{'='*62}")

    s1_name = cfg["stage1"]["model"]["name"]
    s2_name = cfg["stage2"]["model"]["name"]

    s1_model, s1_ckpt = _load_checkpoint(run_dir, "stage1", cfg, device)
    s2_model, s2_ckpt = _load_checkpoint(run_dir, "stage2", cfg, device)

    # Val loaders for threshold search
    _, val_s1, _, _ = get_dataloaders(cfg, stage="stage1")
    _, val_s2, _, _ = get_dataloaders(cfg, stage="stage2")

    vp1, vl1 = _infer(s1_model, val_s1, device, "S1 Val")
    best_t1   = find_best_threshold_binary(
        vp1.flatten(), vl1.flatten(), candidates=np.arange(0.05, 0.95, 0.05), metric="f1"
    )
    vp2, vl2 = _infer(s2_model, val_s2, device, "S2 Val")
    best_ts2  = find_best_thresholds(vp2, vl2, candidates=np.arange(0.05, 0.95, 0.05), metric="f1")
    print(f"[test] S1 threshold: {best_t1:.2f}")
    print(f"[test] S2 thresholds: {[round(float(t),2) for t in best_ts2]}")

    # Build full test loaders
    (_, _, _, _, test_texts, test_labels_6) = get_raw_splits(cfg)
    max_length = int(cfg["data"].get("max_length", 128))
    batch_size = int(cfg["stage1"]["training"].get("batch_size", 32))

    s1_tok = AutoTokenizer.from_pretrained(BACKBONE_REGISTRY[s1_name]["pretrained"])
    s2_tok = AutoTokenizer.from_pretrained(BACKBONE_REGISTRY[s2_name]["pretrained"])

    def _make_loader(tok, stage_name):
        ds = EkmanDataset(test_texts, test_labels_6, tok, max_length,
                          stage=stage_name, emotion_only=False, augment_rare=False)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    s1_probs, _ = _infer(s1_model, _make_loader(s1_tok, "stage1"), device, "E2E Stage1")
    s2_probs, _ = _infer(s2_model, _make_loader(s2_tok, "stage2"), device, "E2E Stage2")

    s1_probs = s1_probs.flatten()
    s1_preds = (s1_probs >= best_t1).astype(int)
    s2_preds = apply_threshold(s2_probs, best_ts2)

    N = len(test_labels_6)
    has_emotion_true = (test_labels_6.sum(axis=1) > 0)

    # Build 7-dim ground-truth
    y_true_7 = np.zeros((N, 7), dtype=np.int32)
    y_true_7[:, :6] = test_labels_6.astype(np.int32)
    y_true_7[~has_emotion_true, 6] = 1

    # Build 7-dim predictions
    y_pred_7 = np.zeros((N, 7), dtype=np.int32)
    for i in range(N):
        if s1_preds[i] == 0:
            y_pred_7[i, 6] = 1                # → neutral
        else:
            em = s2_preds[i]
            if em.sum() == 0:
                y_pred_7[i, 6] = 1            # fallback → neutral
            else:
                y_pred_7[i, :6] = em

    micro_f1    = f1_score(y_true_7, y_pred_7, average="micro",    zero_division=0)
    macro_f1    = f1_score(y_true_7, y_pred_7, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true_7, y_pred_7, average="weighted", zero_division=0)
    hamming     = hamming_loss(y_true_7, y_pred_7)
    subset_acc  = float((y_pred_7 == y_true_7).all(axis=1).mean())
    per_f1      = f1_score(y_true_7, y_pred_7, average=None, zero_division=0)
    per_p       = precision_score(y_true_7, y_pred_7, average=None, zero_division=0)
    per_r       = recall_score(y_true_7, y_pred_7, average=None, zero_division=0)
    per_sup     = y_true_7.sum(axis=0).astype(int)
    report      = classification_report(y_true_7, y_pred_7, target_names=ALL_CLASS_NAMES, zero_division=0)

    s1_acc  = accuracy_score(has_emotion_true.astype(int), s1_preds)
    s1_f1   = f1_score(has_emotion_true.astype(int), s1_preds, zero_division=0)

    print(f"\n{'='*62}")
    print(f"  Micro  F1    : {micro_f1:.4f}")
    print(f"  Macro  F1    : {macro_f1:.4f}")
    print(f"  Weighted F1  : {weighted_f1:.4f}")
    print(f"  Hamming Loss : {hamming:.4f}")
    print(f"  Subset Acc   : {subset_acc:.4f}")
    print(f"  Stage1 Acc   : {s1_acc:.4f}  Stage1 F1: {s1_f1:.4f}  (routing quality)")
    print(f"{'='*62}\n{report}")

    # Save
    with open(os.path.join(out_dir, "e2e_report.txt"), "w") as f:
        f.write(f"End-to-End Pipeline — 7 Classes\n")
        f.write(f"Stage1: {s1_name} (epoch {s1_ckpt.get('epoch','?')})\n")
        f.write(f"Stage2: {s2_name} (epoch {s2_ckpt.get('epoch','?')})\n")
        f.write(f"S1 threshold: {best_t1:.2f}\n")
        f.write(f"S2 thresholds: {[round(float(t),2) for t in best_ts2]}\n\n")
        f.write(f"Micro F1: {micro_f1:.4f}  Macro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}  Hamming: {hamming:.4f}\n")
        f.write(f"Subset Acc: {subset_acc:.4f}\n")
        f.write(f"Stage1 Acc: {s1_acc:.4f}  Stage1 F1: {s1_f1:.4f}\n\n{report}")

    with open(os.path.join(out_dir, "e2e_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "stage1_model","stage2_model","micro_f1","macro_f1","weighted_f1",
            "hamming","subset_acc","stage1_acc","stage1_f1"
        ])
        w.writeheader()
        w.writerow({"stage1_model": s1_name, "stage2_model": s2_name,
                    "micro_f1": round(micro_f1, 4), "macro_f1": round(macro_f1, 4),
                    "weighted_f1": round(weighted_f1, 4), "hamming": round(hamming, 4),
                    "subset_acc": round(subset_acc, 4), "stage1_acc": round(s1_acc, 4),
                    "stage1_f1": round(s1_f1, 4)})

    with open(os.path.join(out_dir, "e2e_per_class.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class","precision","recall","f1","support"])
        w.writeheader()
        for i, name in enumerate(ALL_CLASS_NAMES):
            w.writerow({"class": name, "precision": round(float(per_p[i]), 4),
                        "recall": round(float(per_r[i]), 4),
                        "f1": round(float(per_f1[i]), 4), "support": int(per_sup[i])})

    _plot_hbar_f1(per_f1, ALL_CLASS_NAMES,
                  os.path.join(out_dir, "e2e_per_class_f1.png"),
                  "End-to-End Per-Class F1 (7 classes, full test set)")
    _plot_heatmap(s2_probs, test_labels_6, EMOTION_NAMES,
                  os.path.join(out_dir, "e2e_stage2_heatmap.png"),
                  title="End-to-End Stage2 Probabilities (test subset)")

    # Build 7-dim probability matrix for PR/CM plots
    # cols 0-5 = s2_probs; col 6 = 1 - s1_probs (neutral probability)
    s1_neutral_prob = 1.0 - s1_probs          # higher → more likely neutral
    y_probs_7 = np.concatenate(
        [s2_probs, s1_neutral_prob[:, None]], axis=1
    )
    _plot_pr_curve_multiclass(y_true_7, y_probs_7, ALL_CLASS_NAMES,
                              os.path.join(out_dir, "e2e_pr_curve.png"),
                              "End-to-End Per-Class PR Curves (7 classes, full test set)")
    _plot_confusion_multilabel(y_true_7, y_pred_7, ALL_CLASS_NAMES,
                               os.path.join(out_dir, "e2e_confusion.png"),
                               "End-to-End Per-Class Confusion Matrices (test set)")
    # Also save a single aggregated 7-class confusion matrix
    # (argmax prediction for samples predicted as exactly one class)
    y_true_single = np.argmax(y_true_7, axis=1)
    y_pred_single = np.argmax(y_pred_7, axis=1)
    # Only use rows where ground-truth has exactly one active label
    single_mask = y_true_7.sum(axis=1) == 1
    if single_mask.sum() > 0:
        cm7 = confusion_matrix(y_true_single[single_mask],
                               y_pred_single[single_mask],
                               labels=list(range(7)))
        _plot_confusion(cm7, ALL_CLASS_NAMES,
                        os.path.join(out_dir, "e2e_confusion_aggregate.png"),
                        "End-to-End Aggregated Confusion Matrix (single-label samples)")
    print(f"[test] Saved → {out_dir}")

    return {"micro_f1": micro_f1, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "hamming": hamming, "subset_accuracy": subset_acc,
            "stage1_accuracy": s1_acc, "stage1_f1": s1_f1,
            "best_threshold_s1": best_t1, "best_thresholds_s2": best_ts2,
            "out_dir": out_dir}


# =============================================================================
#  CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    type=str, default="all",
                        choices=["stage1", "stage2", "end2end", "all"])
    parser.add_argument("--config",  type=str, default="config/config.yaml")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Explicit run directory (default: auto-detect latest)")
    args = parser.parse_args()

    kw = {"config_path": args.config, "run_dir": args.run_dir}
    if args.mode in ("stage1", "all"):   evaluate_stage1(**kw)
    if args.mode in ("stage2", "all"):   evaluate_stage2(**kw)
    if args.mode in ("end2end", "all"):  evaluate_end_to_end(**kw)
