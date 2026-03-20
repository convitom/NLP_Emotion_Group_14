"""
src/dataloader.py
Single-Stage End-to-End Ekman Classification — DataLoader.

Task: 7-class multi-label (6 emotions + neutral) on ALL samples.

Data format (CSV):
    text | anger | disgust | fear | joy | sadness | surprise
    - Emotion samples : one or more emotion columns = 1
    - Neutral samples : all 6 emotion columns = 0  →  neutral label auto-derived

Class counts (approx):
    joy      19,794  ┐
    anger     7,813  │ common
    surprise  5,418  │
    sadness   5,392  ┘
    neutral  14,497    rare
    disgust   3,366    rare
    fear      1,941    very_rare

Three-tier imbalance strategy — tiers computed dynamically from train data.
"""

from __future__ import annotations

import os
import random
import re
from typing import Dict, List, Tuple

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
NUM_EMOTIONS:  int        = len(EMOTION_NAMES)    # 6
CLASS_NAMES:   List[str] = EMOTION_NAMES + ["neutral"]
NUM_CLASSES:   int        = len(CLASS_NAMES)      # 7
NEUTRAL_IDX:   int        = 6


# =============================================================================
#  Synonym augmentation
# =============================================================================

_SYNONYM_MAP: Dict[str, List[str]] = {
    "happy":     ["glad", "pleased", "delighted", "joyful"],
    "good":      ["great", "wonderful", "fantastic", "excellent"],
    "bad":       ["terrible", "awful", "horrible", "dreadful"],
    "sad":       ["unhappy", "miserable", "sorrowful", "depressed"],
    "angry":     ["furious", "enraged", "mad", "irritated"],
    "scared":    ["afraid", "frightened", "terrified", "anxious"],
    "love":      ["adore", "cherish", "treasure", "care about"],
    "hate":      ["despise", "loathe", "detest", "dislike"],
    "think":     ["believe", "feel", "consider", "reckon"],
    "amazing":   ["incredible", "awesome", "remarkable", "stunning"],
    "hard":      ["difficult", "tough", "challenging", "demanding"],
    "hope":      ["wish", "expect", "trust", "anticipate"],
    "worry":     ["fear", "dread", "fret", "stress"],
    "thankful":  ["grateful", "appreciative", "blessed"],
    "proud":     ["honored", "pleased", "satisfied", "glad"],
    "awful":     ["terrible", "dreadful", "horrible", "atrocious"],
    "excited":   ["thrilled", "enthusiastic", "eager", "pumped"],
    "surprised": ["shocked", "astonished", "stunned", "amazed"],
    "fearful":   ["terrified", "petrified", "alarmed", "horrified"],
    "disgusted": ["revolted", "repulsed", "sickened", "appalled"],
    "furious":   ["enraged", "livid", "irate", "outraged"],
    "depressed": ["despondent", "devastated", "heartbroken", "gloomy"],
    "joyful":    ["ecstatic", "elated", "overjoyed", "blissful"],
    "nervous":   ["anxious", "apprehensive", "uneasy", "tense"],
    "shocked":   ["stunned", "astounded", "flabbergasted", "aghast"],
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
    label_counts:  np.ndarray,
    very_rare_div: float = 3.0,
    rare_div:      float = 1.0,
) -> Tuple[List[int], List[int], List[int]]:
    median           = float(np.median(label_counts))
    thresh_very_rare = median / very_rare_div
    thresh_rare      = median / rare_div
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
    7-class multi-label dataset for end-to-end Ekman classification.

    Labels: (7,) float32
        cols 0-5 = emotion one-hots  (EMOTION_NAMES order)
        col  6   = neutral            (1 iff all emotion cols = 0)
    """

    def __init__(
        self,
        texts:               List[str],
        labels_6:            np.ndarray,
        tokenizer,
        max_length:          int  = 128,
        augment_rare:        bool = False,
        aug_copies_per_tier: Dict[str, int]       = None,
        tier_indices:        Dict[str, List[int]] = None,
    ) -> None:
        if aug_copies_per_tier is None:
            aug_copies_per_tier = {"very_rare": 4, "rare": 2, "common": 0}
        if tier_indices is None:
            tier_indices = {"very_rare": [], "rare": [], "common": list(range(NUM_CLASSES))}

        self.tokenizer  = tokenizer
        self.max_length = max_length

        has_emotion = (labels_6.sum(axis=1) > 0).astype(np.float32)
        neutral_col = (1.0 - has_emotion).reshape(-1, 1)
        labels_7    = np.concatenate([labels_6, neutral_col], axis=1).astype(np.float32)

        if augment_rare:
            extra_texts:  List[str]        = []
            extra_labels: List[np.ndarray] = []
            for tier_name, copies in aug_copies_per_tier.items():
                if copies <= 0:
                    continue
                tier_idx = tier_indices.get(tier_name, [])
                if not tier_idx:
                    continue
                tier_mask = labels_7[:, tier_idx].sum(axis=1) > 0
                n_tier    = int(tier_mask.sum())
                for i, (t, l) in enumerate(zip(texts, labels_7)):
                    if tier_mask[i]:
                        for k in range(copies):
                            extra_texts.append(_synonym_replace(t, n=2, seed=i * 100 + k))
                            extra_labels.append(l.copy())
                print(f"[DataLoader] Aug tier={tier_name:<10}: "
                      f"{n_tier} samples × {copies} copies → +{n_tier * copies}")
            if extra_texts:
                texts    = list(texts) + extra_texts
                labels_7 = np.vstack([labels_7, np.array(extra_labels)])

        self.texts    = list(texts)
        self.labels_7 = labels_7.astype(np.float32)

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
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels_7[idx], dtype=torch.float32),
        }


# =============================================================================
#  Weighted sampler
# =============================================================================

def build_weighted_sampler(
    dataset:         EkmanDataset,
    sampler_power:   float,
    boost_very_rare: float,
    boost_rare:      float,
    boost_common:    float,
    tier_indices:    Dict[str, List[int]],
) -> WeightedRandomSampler:
    labels_mat   = dataset.labels_7
    label_counts = labels_mat.sum(axis=0).clip(min=1)
    inv_freq     = (1.0 / label_counts) ** sampler_power

    sample_weights = np.zeros(len(dataset), dtype=np.float64)
    for i, row in enumerate(labels_mat):
        pos = row > 0
        sample_weights[i] = inv_freq[pos].mean() if pos.any() else inv_freq.min()

    for tier, boost in {"very_rare": boost_very_rare, "rare": boost_rare, "common": boost_common}.items():
        if boost <= 1.0:
            continue
        idx = tier_indices.get(tier, [])
        if idx:
            sample_weights[labels_mat[:, idx].sum(axis=1) > 0] *= boost

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(dataset),
        replacement=True,
    )


# =============================================================================
#  pos_weight
# =============================================================================

def compute_pos_weight(
    dataset:            EkmanDataset,
    device:             torch.device,
    pw_scale_very_rare: float,
    pw_scale_rare:      float,
    pw_scale_common:    float,
    tier_indices:       Dict[str, List[int]],
) -> torch.Tensor:
    n            = len(dataset)
    labels_mat   = dataset.labels_7
    pos_counts   = (labels_mat > 0).sum(axis=0).astype(np.float64)  # đếm số ROWS có class đó → luôn <= n
    pos_counts   = np.maximum(pos_counts, 1.0)
    neg_counts   = np.maximum(n - pos_counts, 1.0)                   # luôn >= 0
    label_counts = pos_counts        # tránh âm

    scale_map = np.ones(NUM_CLASSES, dtype=np.float32)
    for idx in tier_indices.get("very_rare", []):
        scale_map[idx] = pw_scale_very_rare
    for idx in tier_indices.get("rare", []):
        scale_map[idx] = pw_scale_rare
    for idx in tier_indices.get("common", []):
        scale_map[idx] = pw_scale_common

    pw = (neg_counts / label_counts) * scale_map  # ✅ * thay vì **
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
    return df["text"].astype(str).tolist(), df[EMOTION_NAMES].values.astype(np.float32)


def _split_data(
    texts: List[str], labels_6: np.ndarray,
    val_ratio: float = 0.10, test_ratio: float = 0.10, seed: int = 42,
) -> Tuple:
    strat = (labels_6.sum(axis=1) > 0).astype(int)
    tv_t, test_t, tv_l, test_l = train_test_split(
        texts, labels_6, test_size=test_ratio, random_state=seed, stratify=strat,
    )
    strat2 = (tv_l.sum(axis=1) > 0).astype(int)
    tr_t, val_t, tr_l, val_l = train_test_split(
        tv_t, tv_l,
        test_size=val_ratio / (1.0 - test_ratio), random_state=seed, stratify=strat2,
    )
    return tr_t, tr_l, val_t, val_l, test_t, test_l


# =============================================================================
#  Main factory
# =============================================================================

def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Build train/val/test DataLoaders for 7-class end-to-end task.

    Returns:
        (train_loader, val_loader, test_loader, info_dict)

    info_dict keys: class_names, pos_weight, label_counts,
                    tier_indices, num_labels, pretrained
    """
    data_cfg  = cfg["data"]
    stage_cfg = cfg["e2e"]
    train_cfg = stage_cfg["training"]

    data_dir    = data_cfg["data_dir"]
    max_length  = int(data_cfg.get("max_length", 128))
    batch_size  = int(train_cfg.get("batch_size", 32))
    seed        = int(data_cfg.get("seed", 42))
    auto_split  = bool(data_cfg.get("auto_split", True))
    num_workers = int(data_cfg.get("num_workers", 2))

    model_name = stage_cfg["model"]["name"].lower()
    if model_name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {' | '.join(BACKBONE_REGISTRY)}")
    pretrained = BACKBONE_REGISTRY[model_name]["pretrained"]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)

    # ── Load raw data ─────────────────────────────────────────────────────────
    train_path = os.path.join(data_dir, data_cfg.get("train_file", "data1_train.csv"))
    val_path   = os.path.join(data_dir, data_cfg.get("val_file",   "data1_val.csv"))
    test_path  = os.path.join(data_dir, data_cfg.get("test_file",  "data1_test.csv"))

    if auto_split and (not os.path.isfile(val_path) or not os.path.isfile(test_path)):
        print(f"[DataLoader] Auto-splitting '{train_path}'")
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

    # ── Tier computation on 7-class train counts ──────────────────────────────
    train_neutral_col = (~(train_labels.sum(axis=1) > 0)).astype(np.float32).reshape(-1, 1)
    train_labels_7    = np.concatenate([train_labels, train_neutral_col], axis=1)
    train_counts_7    = train_labels_7.sum(axis=0)

    very_rare_div = float(train_cfg.get("very_rare_divisor", 3.0))
    rare_div      = float(train_cfg.get("rare_divisor",      1.0))
    very_rare_idx, rare_idx, common_idx = compute_tiers(train_counts_7, very_rare_div, rare_div)
    tier_indices = {"very_rare": very_rare_idx, "rare": rare_idx, "common": common_idx}

    print(f"[DataLoader] backbone={model_name}")
    print(f"[DataLoader] Tier breakdown (train counts):")
    for i, name in enumerate(CLASS_NAMES):
        tier = ("very_rare" if i in very_rare_idx else "rare" if i in rare_idx else "common")
        print(f"    {name:<12}: {int(train_counts_7[i]):>6}  [{tier}]")

    augment_rare = bool(train_cfg.get("augment_rare", False))
    aug_copies   = {
        "very_rare": int(train_cfg.get("aug_copies_very_rare", 4)),
        "rare":      int(train_cfg.get("aug_copies_rare",      2)),
        "common":    int(train_cfg.get("aug_copies_common",    0)),
    }

    train_ds = EkmanDataset(train_texts, train_labels, tokenizer, max_length,
                            augment_rare=augment_rare, aug_copies_per_tier=aug_copies,
                            tier_indices=tier_indices)
    val_ds   = EkmanDataset(val_texts,   val_labels,   tokenizer, max_length)
    test_ds  = EkmanDataset(test_texts,  test_labels,  tokenizer, max_length)

    use_sampler = bool(train_cfg.get("use_weighted_sampler", True))
    if use_sampler:
        sampler = build_weighted_sampler(
            train_ds,
            sampler_power=   float(train_cfg.get("sampler_power",   2.0)),
            boost_very_rare= float(train_cfg.get("boost_very_rare", 5.0)),
            boost_rare=      float(train_cfg.get("boost_rare",      3.0)),
            boost_common=    float(train_cfg.get("boost_common",    1.0)),
            tier_indices=tier_indices,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True,
                                  persistent_workers=(num_workers > 0))
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True,
                                  persistent_workers=(num_workers > 0))

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,persistent_workers=(num_workers > 0))

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = compute_pos_weight(
        train_ds, device,
        pw_scale_very_rare=float(train_cfg.get("pw_scale_very_rare", 2.0)),
        pw_scale_rare=     float(train_cfg.get("pw_scale_rare",      1.5)),
        pw_scale_common=   float(train_cfg.get("pw_scale_common",    1.0)),
        tier_indices=tier_indices,
    )

    label_counts_dict = {CLASS_NAMES[i]: int(train_ds.labels_7[:, i].sum()) for i in range(NUM_CLASSES)}
    print(f"[DataLoader] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"[DataLoader] pos_weight:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:<12}: {pos_weight[i].item():.2f}")

    return train_loader, val_loader, test_loader, {
        "class_names":  CLASS_NAMES,
        "pos_weight":   pos_weight,
        "label_counts": label_counts_dict,
        "tier_indices": tier_indices,
        "num_labels":   NUM_CLASSES,
        "pretrained":   pretrained,
    }
