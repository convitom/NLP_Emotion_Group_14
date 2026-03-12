"""
src/dataloader.py
Two-Stage Ekman Emotion Classification — DataLoader.

Data format (CSV):
    text | anger | disgust | fear | joy | sadness | surprise

Stage modes:
    stage="stage1"  → binary label [has_emotion]  — all samples
    stage="stage2"  → 6-hot label  [emotions]     — emotion-only samples

Three-tier imbalance strategy for Stage 2
──────────────────────────────────────────
Class counts in data1_train.csv:
  joy      19,794   ┐
  anger     7,813   │ common  (count ≥ median)
  surprise  5,418   ┘
  sadness   5,392   ┐
  disgust   3,366   ┘ rare    (median/3 ≤ count < median)
  fear      1,941     very_rare (count < median/3)

  joy/fear ratio ≈ 10×  →  three separate treatment tiers:

  Layer 1 – Augmentation:
      very_rare: aug_copies_very_rare (default 4) synonym-replaced copies
      rare:      aug_copies_rare      (default 2)
      common:    aug_copies_common    (default 0)

  Layer 2 – WeightedRandomSampler:
      inv_freq^sampler_power  +  per-tier boost multiplier
      very_rare: boost_very_rare (default 5×)
      rare:      boost_rare      (default 3×)

  Layer 3 – pos_weight:
      per-tier pw_scale exponent applied to ((N-n_pos)/n_pos)
      very_rare: pw_scale_very_rare (default 2.0)
      rare:      pw_scale_rare      (default 1.5)
      common:    pw_scale_common    (default 1.0)
"""

from __future__ import annotations

import os
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer


# =============================================================================
#  Backbone registry
# =============================================================================

BACKBONE_REGISTRY: Dict[str, Dict[str, str]] = {
    "bert":    {"pretrained": "google-bert/bert-base-uncased",     "amp_dtype": "float16"},
    "roberta": {"pretrained": "FacebookAI/roberta-base",           "amp_dtype": "float16"},
    "deberta": {"pretrained": "microsoft/deberta-v3-base",         "amp_dtype": "bfloat16"},
    "electra": {"pretrained": "google/electra-base-discriminator", "amp_dtype": "float16"},
}

# =============================================================================
#  Label metadata
# =============================================================================

EMOTION_NAMES: List[str] = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
NUM_EMOTIONS:  int        = len(EMOTION_NAMES)   # 6
ALL_CLASS_NAMES: List[str] = EMOTION_NAMES + ["neutral"]
NUM_ALL_CLASSES: int        = 7


# =============================================================================
#  Synonym augmentation
# =============================================================================

_SYNONYM_MAP: Dict[str, List[str]] = {
    "happy":       ["glad", "pleased", "delighted", "joyful"],
    "good":        ["great", "wonderful", "fantastic", "excellent"],
    "bad":         ["terrible", "awful", "horrible", "dreadful"],
    "sad":         ["unhappy", "miserable", "sorrowful", "depressed"],
    "angry":       ["furious", "enraged", "mad", "irritated"],
    "scared":      ["afraid", "frightened", "terrified", "anxious"],
    "love":        ["adore", "cherish", "treasure", "care about"],
    "hate":        ["despise", "loathe", "detest", "dislike"],
    "think":       ["believe", "feel", "consider", "reckon"],
    "amazing":     ["incredible", "awesome", "remarkable", "stunning"],
    "hard":        ["difficult", "tough", "challenging", "demanding"],
    "hope":        ["wish", "expect", "trust", "anticipate"],
    "worry":       ["fear", "dread", "fret", "stress"],
    "thankful":    ["grateful", "appreciative", "blessed"],
    "proud":       ["honored", "pleased", "satisfied", "glad"],
    "awful":       ["terrible", "dreadful", "horrible", "atrocious"],
    "excited":     ["thrilled", "enthusiastic", "eager", "pumped"],
    "surprised":   ["shocked", "astonished", "stunned", "amazed"],
    "fearful":     ["terrified", "petrified", "alarmed", "horrified"],
    "disgusted":   ["revolted", "repulsed", "sickened", "appalled"],
    "furious":     ["enraged", "livid", "irate", "outraged"],
    "depressed":   ["despondent", "devastated", "heartbroken", "gloomy"],
    "joyful":      ["ecstatic", "elated", "overjoyed", "blissful"],
    "nervous":     ["anxious", "apprehensive", "uneasy", "tense"],
    "shocked":     ["stunned", "astounded", "flabbergasted", "aghast"],
}


def _synonym_replace(text: str, n: int = 2, seed: int = None) -> str:
    rng   = random.Random(seed)
    words = text.split()
    idxs  = list(range(len(words)))
    rng.shuffle(idxs)
    replaced = 0
    for i in idxs:
        w = re.sub(r"[^a-zA-Z]", "", words[i]).lower()
        if w in _SYNONYM_MAP and replaced < n:
            words[i] = rng.choice(_SYNONYM_MAP[w])
            replaced += 1
    return " ".join(words)


# =============================================================================
#  Tier classification
# =============================================================================

def compute_tiers(
    label_counts:     np.ndarray,
    very_rare_div:    float = 3.0,
    rare_div:         float = 1.0,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Classify class indices into three tiers based on count vs median.

    Args:
        label_counts:  (C,) array of per-class positive counts.
        very_rare_div: threshold = median / very_rare_div
        rare_div:      threshold = median / rare_div  (= median when rare_div=1)

    Returns:
        (very_rare_indices, rare_indices, common_indices)
    """
    median = float(np.median(label_counts))
    thresh_very_rare = median / very_rare_div
    thresh_rare      = median / rare_div       # = median

    very_rare, rare, common = [], [], []
    for i, c in enumerate(label_counts):
        if c < thresh_very_rare:
            very_rare.append(i)
        elif c < thresh_rare:
            rare.append(i)
        else:
            common.append(i)

    return very_rare, rare, common


# =============================================================================
#  Dataset
# =============================================================================

class EkmanDataset(Dataset):
    """
    Dataset for two-stage Ekman classification.

    Args:
        texts:              List[str]
        labels_6:           (N, 6) float32 — one-hot 6 Ekman emotions
        tokenizer:          HuggingFace tokenizer
        max_length:         int
        stage:              "stage1" → binary label (1,)
                            "stage2" → 6-hot label (6,)
        emotion_only:       Filter to emotion-positive samples (stage2 training)
        augment_rare:       Apply synonym replacement augmentation
        aug_copies_per_tier: dict with keys "very_rare", "rare", "common" → int
        tier_indices:       dict with keys "very_rare", "rare", "common" → List[int]
    """

    def __init__(
        self,
        texts:               List[str],
        labels_6:            np.ndarray,
        tokenizer:           AutoTokenizer,
        max_length:          int  = 128,
        stage:               str  = "stage2",
        emotion_only:        bool = False,
        augment_rare:        bool = False,
        aug_copies_per_tier: Dict[str, int]       = None,
        tier_indices:        Dict[str, List[int]] = None,
    ) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.stage      = stage

        if aug_copies_per_tier is None:
            aug_copies_per_tier = {"very_rare": 4, "rare": 2, "common": 0}
        if tier_indices is None:
            tier_indices = {"very_rare": [], "rare": [], "common": list(range(NUM_EMOTIONS))}

        has_emotion = (labels_6.sum(axis=1) > 0).astype(np.float32)

        # Filter to emotion-only for stage2 training
        if emotion_only:
            mask     = has_emotion.astype(bool)
            texts    = [t for t, m in zip(texts, mask) if m]
            labels_6 = labels_6[mask]
            has_emotion = has_emotion[mask]

        # Augmentation (stage2 only)
        if augment_rare and stage == "stage2":
            extra_texts:  List[str]        = []
            extra_labels: List[np.ndarray] = []

            for tier_name, copies in aug_copies_per_tier.items():
                if copies <= 0:
                    continue
                tier_idx = tier_indices.get(tier_name, [])
                if not tier_idx:
                    continue
                tier_mask = labels_6[:, tier_idx].sum(axis=1) > 0
                n_tier    = int(tier_mask.sum())
                for i, (t, l) in enumerate(zip(texts, labels_6)):
                    if tier_mask[i]:
                        for k in range(copies):
                            aug = _synonym_replace(t, n=2, seed=i * 100 + k)
                            extra_texts.append(aug)
                            extra_labels.append(l.copy())
                print(f"[DataLoader] Aug tier={tier_name:<10}: "
                      f"{n_tier} samples × {copies} copies → +{n_tier * copies}")

            if extra_texts:
                texts    = list(texts) + extra_texts
                labels_6 = np.vstack([labels_6, np.array(extra_labels)])
                has_emotion = (labels_6.sum(axis=1) > 0).astype(np.float32)

        self.texts       = list(texts)
        self.labels_6    = labels_6.astype(np.float32)
        self.has_emotion = has_emotion

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.stage == "stage1":
            item["labels"] = torch.tensor([self.has_emotion[idx]], dtype=torch.float32)
        else:
            item["labels"] = torch.tensor(self.labels_6[idx],      dtype=torch.float32)
        return item


# =============================================================================
#  Weighted sampler — three-tier
# =============================================================================

def build_weighted_sampler(
    dataset:      EkmanDataset,
    sampler_power: float,
    boost_very_rare: float,
    boost_rare:      float,
    boost_common:    float,
    tier_indices:   Dict[str, List[int]],
) -> WeightedRandomSampler:
    """Three-tier WeightedRandomSampler for Stage 2."""
    labels_mat   = dataset.labels_6
    label_counts = labels_mat.sum(axis=0).clip(min=1)
    inv_freq     = (1.0 / label_counts) ** sampler_power    # (6,)

    # Base weight = mean of inv_freq for the classes present in that sample
    sample_weights = np.zeros(len(dataset), dtype=np.float64)
    for i, row in enumerate(labels_mat):
        pos = row > 0
        sample_weights[i] = inv_freq[pos].mean() if pos.any() else inv_freq.min()

    # Per-tier boost
    boost_map = {
        "very_rare": boost_very_rare,
        "rare":      boost_rare,
        "common":    boost_common,
    }
    for tier, boost in boost_map.items():
        if boost <= 1.0:
            continue
        idx = tier_indices.get(tier, [])
        if not idx:
            continue
        mask = labels_mat[:, idx].sum(axis=1) > 0
        sample_weights[mask] *= boost

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(dataset),
        replacement=True,
    )


# =============================================================================
#  pos_weight helpers
# =============================================================================

def compute_pos_weight_stage1(
    dataset: EkmanDataset,
    device:  torch.device,
    scale:   float = 1.0,
) -> torch.Tensor:
    """(1,) pos_weight for Stage 1 binary BCE."""
    n_emotion = float(dataset.has_emotion.sum().clip(min=1))
    n_neutral = float(len(dataset) - n_emotion)
    pw = (n_neutral / n_emotion) ** scale
    return torch.tensor([pw], dtype=torch.float32, device=device)


def compute_pos_weight_stage2(
    dataset:            EkmanDataset,
    device:             torch.device,
    pw_scale_very_rare: float,
    pw_scale_rare:      float,
    pw_scale_common:    float,
    tier_indices:       Dict[str, List[int]],
) -> torch.Tensor:
    """
    (6,) per-class pos_weight for Stage 2.
    Each class gets a tier-specific pw_scale exponent.
    """
    n            = len(dataset)
    label_counts = dataset.labels_6.sum(axis=0).clip(min=1)

    scale_map = np.ones(NUM_EMOTIONS, dtype=np.float32)
    for idx in tier_indices.get("very_rare", []):
        scale_map[idx] = pw_scale_very_rare
    for idx in tier_indices.get("rare", []):
        scale_map[idx] = pw_scale_rare
    for idx in tier_indices.get("common", []):
        scale_map[idx] = pw_scale_common

    pw = ((n - label_counts) / label_counts) ** scale_map
    return torch.tensor(pw, dtype=torch.float32, device=device)


# =============================================================================
#  CSV loading & splitting
# =============================================================================

def _load_csv(filepath: str) -> Tuple[List[str], np.ndarray]:
    df      = pd.read_csv(filepath)
    missing = [c for c in EMOTION_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {filepath}")
    if "text" not in df.columns:
        raise ValueError(f"'text' column not found in {filepath}")
    texts    = df["text"].astype(str).tolist()
    labels_6 = df[EMOTION_NAMES].values.astype(np.float32)
    return texts, labels_6


def _split_data(
    texts:      List[str],
    labels_6:   np.ndarray,
    val_ratio:  float = 0.10,
    test_ratio: float = 0.10,
    seed:       int   = 42,
) -> Tuple:
    """Stratified split preserving has_emotion balance."""
    strat = (labels_6.sum(axis=1) > 0).astype(int)
    tv_t, test_t, tv_l, test_l = train_test_split(
        texts, labels_6, test_size=test_ratio, random_state=seed,
        stratify=strat,
    )
    strat2 = (tv_l.sum(axis=1) > 0).astype(int)
    tr_t, val_t, tr_l, val_l = train_test_split(
        tv_t, tv_l,
        test_size=val_ratio / (1.0 - test_ratio), random_state=seed,
        stratify=strat2,
    )
    return tr_t, tr_l, val_t, val_l, test_t, test_l


# =============================================================================
#  Main factory
# =============================================================================

def get_dataloaders(
    cfg:   dict,
    stage: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Build train/val/test DataLoaders for the given stage.

    Returns:
        (train_loader, val_loader, test_loader, info_dict)

    info_dict keys: emotion_names, pos_weight, label_counts,
                    tier_indices, num_labels, pretrained
    """
    assert stage in ("stage1", "stage2"), f"Unknown stage '{stage}'"

    data_cfg  = cfg["data"]
    stage_cfg = cfg[stage]
    train_cfg = stage_cfg["training"]

    data_dir     = data_cfg["data_dir"]
    max_length   = int(data_cfg.get("max_length", 128))
    batch_size   = int(train_cfg.get("batch_size", 32))
    seed         = int(data_cfg.get("seed", 42))
    auto_split   = bool(data_cfg.get("auto_split", True))
    num_workers  = int(data_cfg.get("num_workers", 4))

    model_name = stage_cfg["model"]["name"].lower()
    if model_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {' | '.join(BACKBONE_REGISTRY.keys())}"
        )
    pretrained = BACKBONE_REGISTRY[model_name]["pretrained"]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)

    # ── Load raw data ─────────────────────────────────────────────────────────
    train_path = os.path.join(data_dir, data_cfg.get("train_file", "data1_train.csv"))
    val_path   = os.path.join(data_dir, data_cfg.get("val_file",   "data1_val.csv"))
    test_path  = os.path.join(data_dir, data_cfg.get("test_file",  "data1_test.csv"))

    if auto_split and (not os.path.isfile(val_path) or not os.path.isfile(test_path)):
        print(f"[DataLoader] Auto-splitting '{train_path}'  "
              f"→ train/{data_cfg.get('val_ratio',0.1):.0%}/test {data_cfg.get('test_ratio',0.1):.0%}")
        all_t, all_l = _load_csv(train_path)
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
            _split_data(all_t, all_l,
                        val_ratio=float(data_cfg.get("val_ratio",  0.10)),
                        test_ratio=float(data_cfg.get("test_ratio", 0.10)),
                        seed=seed)
    else:
        for split, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"[DataLoader] {split} file not found: '{path}'")
        train_texts, train_labels = _load_csv(train_path)
        val_texts,   val_labels   = _load_csv(val_path)
        test_texts,  test_labels  = _load_csv(test_path)

    # ── Compute tiers from TRAIN label counts ─────────────────────────────────
    train_counts   = train_labels.sum(axis=0)
    very_rare_div  = float(train_cfg.get("very_rare_divisor", 3.0))
    rare_div       = float(train_cfg.get("rare_divisor",      1.0))
    very_rare_idx, rare_idx, common_idx = compute_tiers(
        train_counts, very_rare_div, rare_div
    )
    tier_indices = {
        "very_rare": very_rare_idx,
        "rare":      rare_idx,
        "common":    common_idx,
    }
    print(f"[DataLoader] Stage={stage}  backbone={model_name}")
    print(f"[DataLoader] Tier breakdown (train counts):")
    for i, name in enumerate(EMOTION_NAMES):
        tier = ("very_rare" if i in very_rare_idx else
                "rare"      if i in rare_idx else "common")
        print(f"    {name:<12}: {int(train_counts[i]):>6}  [{tier}]")

    # ── Stage-specific dataset config ────────────────────────────────────────
    emotion_only_train = (stage == "stage2")
    augment_rare       = bool(train_cfg.get("augment_rare", False))
    aug_copies = {
        "very_rare": int(train_cfg.get("aug_copies_very_rare", 4)),
        "rare":      int(train_cfg.get("aug_copies_rare",      2)),
        "common":    int(train_cfg.get("aug_copies_common",    0)),
    }

    # ── Build datasets ────────────────────────────────────────────────────────
    train_ds = EkmanDataset(
        train_texts, train_labels, tokenizer, max_length,
        stage=stage, emotion_only=emotion_only_train,
        augment_rare=augment_rare, aug_copies_per_tier=aug_copies,
        tier_indices=tier_indices,
    )
    val_ds = EkmanDataset(
        val_texts, val_labels, tokenizer, max_length,
        stage=stage, emotion_only=emotion_only_train,
        augment_rare=False,
    )
    test_ds = EkmanDataset(
        test_texts, test_labels, tokenizer, max_length,
        stage=stage, emotion_only=emotion_only_train,
        augment_rare=False,
    )

    # ── Sampler ───────────────────────────────────────────────────────────────
    use_sampler = bool(train_cfg.get("use_weighted_sampler", stage == "stage2"))

    if use_sampler and stage == "stage2":
        sampler = build_weighted_sampler(
            train_ds,
            sampler_power=float(train_cfg.get("sampler_power", 2.0)),
            boost_very_rare=float(train_cfg.get("boost_very_rare", 5.0)),
            boost_rare=     float(train_cfg.get("boost_rare",      3.0)),
            boost_common=   float(train_cfg.get("boost_common",    1.0)),
            tier_indices=tier_indices,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ── pos_weight ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if stage == "stage1":
        pw_scale  = float(train_cfg.get("pw_scale", 1.0))
        pos_weight = compute_pos_weight_stage1(train_ds, device, scale=pw_scale)
        num_labels = 1
    else:
        pos_weight = compute_pos_weight_stage2(
            train_ds, device,
            pw_scale_very_rare=float(train_cfg.get("pw_scale_very_rare", 2.0)),
            pw_scale_rare=     float(train_cfg.get("pw_scale_rare",      1.5)),
            pw_scale_common=   float(train_cfg.get("pw_scale_common",    1.0)),
            tier_indices=tier_indices,
        )
        num_labels = NUM_EMOTIONS

    label_counts_dict = {EMOTION_NAMES[i]: int(train_ds.labels_6[:, i].sum())
                         for i in range(NUM_EMOTIONS)}
    ne = int(train_ds.has_emotion.sum())
    label_counts_dict.update({"has_emotion": ne, "neutral": len(train_ds) - ne})

    print(f"[DataLoader] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    if stage == "stage2":
        print(f"[DataLoader] pos_weight:")
        for i, name in enumerate(EMOTION_NAMES):
            print(f"    {name:<12}: {pos_weight[i].item():.2f}")

    return train_loader, val_loader, test_loader, {
        "emotion_names": EMOTION_NAMES,
        "pos_weight":    pos_weight,
        "label_counts":  label_counts_dict,
        "tier_indices":  tier_indices,
        "num_labels":    num_labels,
        "pretrained":    pretrained,
    }


def get_raw_splits(cfg: dict) -> Tuple:
    """Return (train_t, train_l, val_t, val_l, test_t, test_l) without building Datasets."""
    data_cfg   = cfg["data"]
    data_dir   = data_cfg["data_dir"]
    auto_split = bool(data_cfg.get("auto_split", True))
    seed       = int(data_cfg.get("seed", 42))

    train_path = os.path.join(data_dir, data_cfg.get("train_file", "data1_train.csv"))
    val_path   = os.path.join(data_dir, data_cfg.get("val_file",   "data1_val.csv"))
    test_path  = os.path.join(data_dir, data_cfg.get("test_file",  "data1_test.csv"))

    if auto_split and (not os.path.isfile(val_path) or not os.path.isfile(test_path)):
        all_t, all_l = _load_csv(train_path)
        return _split_data(all_t, all_l,
                           val_ratio=float(data_cfg.get("val_ratio",  0.10)),
                           test_ratio=float(data_cfg.get("test_ratio", 0.10)),
                           seed=seed)
    else:
        tr_t, tr_l = _load_csv(train_path)
        va_t, va_l = _load_csv(val_path)
        te_t, te_l = _load_csv(test_path)
        return tr_t, tr_l, va_t, va_l, te_t, te_l
