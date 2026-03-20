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
  surprise  5,418   │
  sadness   5,392   ┘
  disgust   3,366     rare    (median/3 ≤ count < median)
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
      per-tier pw_scale multiplier applied to ((N-n_pos)/n_pos)
      very_rare: pw_scale_very_rare (default 3.0)  ← multiply, NOT exponent
      rare:      pw_scale_rare      (default 1.5)
      common:    pw_scale_common    (default 1.0)

num_workers:
    Set via config['data']['num_workers'] (default 4).
    Pass 0 to disable multiprocessing (useful for debugging or Windows).

Imbalance strategy for Stage 1 (binary: has_emotion vs neutral)
────────────────────────────────────────────────────────────────
Three mutually-exclusive options, controlled by config stage1.training:

  Option A — downsample majority (default off):
      downsample_majority: true
      downsample_ratio:    1.0   # 1.0=1:1, 2.0=2:1, ...
      Pro: mỗi sample gốc giữ nguyên, không mất thông tin thống kê
      Con: dataset nhỏ hơn → ít epochs thực sự

  Option B — oversample minority (default off):
      oversample_minority: true
      oversample_ratio:    1.0   # 1.0=1:1, 2.0=2:1, ...
      Pro: giữ toàn bộ data gốc, thêm copies của minority
      Con: minority bị repeat → risk overfit trên minority

  Option C — WeightedRandomSampler (default off):
      use_weighted_sampler: true
      Pro: không thay đổi dataset, re-weight mỗi epoch
      Con: majority vẫn xuất hiện đủ, model thấy đủ diversity

  Có thể kết hợp B + C để boost mạnh hơn.
  Không nên kết hợp A + B (redundant).
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

EMOTION_NAMES:   List[str] = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
NUM_EMOTIONS:    int        = len(EMOTION_NAMES)   # 6
ALL_CLASS_NAMES: List[str] = EMOTION_NAMES + ["neutral"]
NUM_ALL_CLASSES: int        = 7


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
#  Stage 1 resampling helpers  [MOI THEM]
# =============================================================================

def _resample_stage1(
    texts:       List[str],
    labels_6:    np.ndarray,
    has_emotion: np.ndarray,
    mode:        str,
    ratio:       float,
    seed:        int = 42,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Resample stage1 data de can bang has_emotion vs neutral.
    Dataset-agnostic: tu phat hien class nao la da so / thieu so.

    Args:
        mode : "downsample" bo bot samples cua class da so.
               "oversample" duplicate minority den khi dat ratio.
        ratio: ti le da so / thieu so sau resampling.
               1.0 -> 1:1 balanced | 2.0 -> 2:1 | 0.0 -> tat

    Returns:
        (texts, labels_6, has_emotion) sau resampling.
    """
    if ratio <= 0.0:
        return texts, labels_6, has_emotion

    texts       = list(texts)
    n_emotion   = int(has_emotion.sum())
    n_neutral   = int(len(has_emotion) - n_emotion)
    rng         = np.random.default_rng(seed)

    emotion_idx = np.where(has_emotion > 0)[0]
    neutral_idx = np.where(has_emotion == 0)[0]

    if n_emotion >= n_neutral:
        majority_idx, minority_idx = emotion_idx, neutral_idx
        majority_label = "emotion"
    else:
        majority_idx, minority_idx = neutral_idx, emotion_idx
        majority_label = "neutral"

    n_min = len(minority_idx)
    n_maj = len(majority_idx)

    if mode == "downsample":
        keep_n       = min(int(n_min * ratio), n_maj)
        kept_maj_idx = rng.choice(majority_idx, size=keep_n, replace=False)
        final_idx    = np.sort(np.concatenate([kept_maj_idx, minority_idx]))
        texts_out    = [texts[i]  for i in final_idx]
        labels_out   = labels_6[final_idx]
        has_em_out   = has_emotion[final_idx]

    elif mode == "oversample":
        target_min   = min(int(n_maj * ratio), n_maj * 10)
        extra_needed = max(0, target_min - n_min)
        if extra_needed == 0:
            return texts, labels_6, has_emotion
        dup_idx      = rng.choice(minority_idx, size=extra_needed, replace=True)
        extra_texts  = [texts[i]  for i in dup_idx]
        extra_labels = labels_6[dup_idx]
        extra_has_em = has_emotion[dup_idx]
        texts_out    = texts + extra_texts
        labels_out   = np.vstack([labels_6,    extra_labels])
        has_em_out   = np.concatenate([has_emotion, extra_has_em])

    else:
        raise ValueError(f"mode must be 'downsample' or 'oversample', got '{mode}'")

    n_new_emotion = int(has_em_out.sum())
    n_new_neutral = len(has_em_out) - n_new_emotion
    print(f"[DataLoader] stage1 {mode} (ratio={ratio}, majority={majority_label}): "
          f"{len(has_emotion)} -> {len(has_em_out)} samples  "
          f"(emotion={n_new_emotion}, neutral={n_new_neutral})")
    return texts_out, labels_out, has_em_out


# =============================================================================
#  Tier classification
# =============================================================================

def compute_tiers(
    label_counts:  np.ndarray,
    very_rare_div: float = 3.0,
    rare_div:      float = 1.0,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Classify class indices into three tiers based on count vs median.

    Args:
        label_counts:  (C,) array of per-class positive counts.
        very_rare_div: very_rare threshold = median / very_rare_div
        rare_div:      rare threshold      = median / rare_div  (= median when 1.0)

    Returns:
        (very_rare_indices, rare_indices, common_indices)
    """
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
    Dataset for two-stage Ekman classification.

    Args:
        texts:               List[str]
        labels_6:            (N, 6) float32 — one-hot 6 Ekman emotions
        tokenizer:           HuggingFace tokenizer
        max_length:          int
        stage:               "stage1" → binary label (1,)
                             "stage2" → 6-hot label (6,)
        emotion_only:        Filter to emotion-positive samples (stage2 training)
        augment_rare:        Apply synonym replacement augmentation
        aug_copies_per_tier: dict with keys "very_rare", "rare", "common" → int
        tier_indices:        dict with keys "very_rare", "rare", "common" → List[int]
    """

    def __init__(
        self,
        texts:               List[str],
        labels_6:            np.ndarray,
        tokenizer,
        max_length:          int  = 128,
        stage:               str  = "stage2",
        emotion_only:        bool  = False,
        augment_rare:        bool  = False,
        aug_copies_per_tier: Dict[str, int]       = None,
        tier_indices:        Dict[str, List[int]] = None,
        # [MOI] stage1 resampling ─────────────────────────────────────────
        resample_mode:       str   = "",    # ""=off | "downsample" | "oversample"
        resample_ratio:      float = 1.0,   # target majority:minority ratio
        resample_seed:       int   = 42,
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

        # [MOI] Resample stage1: downsample hoac oversample tren data goc
        # (truoc augmentation de khong bị lẫn copies vào resampling)
        if stage == "stage1" and resample_mode in ("downsample", "oversample"):
            texts, labels_6, has_emotion = _resample_stage1(
                texts, labels_6, has_emotion,
                mode=resample_mode, ratio=resample_ratio, seed=resample_seed,
            )

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
                texts       = list(texts) + extra_texts
                labels_6    = np.vstack([labels_6, np.array(extra_labels)])
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
    dataset:         EkmanDataset,
    sampler_power:   float,
    boost_very_rare: float,
    boost_rare:      float,
    boost_common:    float,
    tier_indices:    Dict[str, List[int]],
) -> WeightedRandomSampler:
    """Three-tier WeightedRandomSampler for Stage 2."""
    labels_mat   = dataset.labels_6
    label_counts = labels_mat.sum(axis=0).clip(min=1)
    inv_freq     = (1.0 / label_counts) ** sampler_power   # (6,)

    # Base weight = mean of inv_freq for the classes present in that sample
    sample_weights = np.zeros(len(dataset), dtype=np.float64)
    for i, row in enumerate(labels_mat):
        pos = row > 0
        sample_weights[i] = inv_freq[pos].mean() if pos.any() else inv_freq.min()

    # Per-tier boost
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
#  pos_weight helpers
# =============================================================================

def compute_pos_weight_stage1(
    dataset: EkmanDataset,
    device:  torch.device,
    scale:   float = 1.0,
) -> torch.Tensor:
    """(1,) pos_weight for Stage 1 binary BCE."""
    n_emotion = max(float(dataset.has_emotion.sum()), 1.0)
    n_neutral = max(float(len(dataset) - n_emotion), 1.0)
    pw = (n_neutral / n_emotion) * scale
    pw = float(np.clip(pw, 0.01, 200.0))
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

    Formula: pw[c] = (neg_count[c] / pos_count[c]) * scale[c]
    where scale[c] is the tier-specific multiplier.

    NOTE: uses multiply (*), NOT exponent (**), to keep values stable.

    IMPORTANT: for multi-label data, neg_count[c] = N - pos_count[c] per class,
    NOT N - total_label_counts. This avoids neg_counts going negative when a
    sample carries multiple emotion labels (which is common after augmentation).
    """
    n            = len(dataset)
    # pos_count[c]  = number of samples where class c == 1
    # neg_count[c]  = number of samples where class c == 0
    # These are per-class and always sum to N, so neg_count is always >= 0.
    labels_mat   = dataset.labels_6
    pos_counts   = (labels_mat > 0).sum(axis=0).astype(np.float64)  # đếm số ROWS có class đó → luôn <= n
    pos_counts   = np.maximum(pos_counts, 1.0)
    neg_counts   = np.maximum(n - pos_counts, 1.0)                   # luôn >= 0
    label_counts = pos_counts                  # (C,) true per-class neg counts

    scale_map = np.ones(NUM_EMOTIONS, dtype=np.float32)
    for idx in tier_indices.get("very_rare", []):
        scale_map[idx] = pw_scale_very_rare
    for idx in tier_indices.get("rare", []):
        scale_map[idx] = pw_scale_rare
    for idx in tier_indices.get("common", []):
        scale_map[idx] = pw_scale_common

    pw = (neg_counts / label_counts) * scale_map        # ← multiply, NOT **
    pw = np.clip(pw, 0.01, 200.0)                       # safety clamp

    print(f"[DataLoader] pos_weight (stage2):")
    for i, name in enumerate(EMOTION_NAMES):
        print(f"    {name:<12}: {pw[i]:.2f}")

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
    texts:      List[str],
    labels_6:   np.ndarray,
    val_ratio:  float = 0.10,
    test_ratio: float = 0.10,
    seed:       int   = 42,
) -> Tuple:
    """Stratified split preserving has_emotion balance."""
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
#  DataLoader factory
# =============================================================================

def _make_loader(
    dataset:     EkmanDataset,
    batch_size:  int,
    shuffle:     bool,
    num_workers: int,
    sampler=None,
) -> DataLoader:
    """Helper to build a DataLoader with consistent settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


def get_dataloaders(
    cfg:   dict,
    stage: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Build train/val/test DataLoaders for the given stage.

    num_workers is read from cfg['data']['num_workers'] (default 4).
    Set to 0 in your config to disable multiprocessing (useful for debugging).

    Returns:
        (train_loader, val_loader, test_loader, info_dict)

    info_dict keys:
        emotion_names, pos_weight, label_counts, tier_indices, num_labels, pretrained
    """
    assert stage in ("stage1", "stage2"), f"Unknown stage '{stage}'"

    data_cfg  = cfg["data"]
    stage_cfg = cfg[stage]
    train_cfg = stage_cfg["training"]

    data_dir    = data_cfg["data_dir"]
    max_length  = int(data_cfg.get("max_length",  128))
    batch_size  = int(train_cfg.get("batch_size", 32))
    seed        = int(data_cfg.get("seed",        42))
    auto_split  = bool(data_cfg.get("auto_split", True))
    # ── num_workers from config — set 0 to disable multiprocessing ───────────
    num_workers = int(data_cfg.get("num_workers", 4))

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
        print(f"[DataLoader] Auto-splitting '{train_path}' "
              f"→ val={data_cfg.get('val_ratio', 0.1):.0%} / "
              f"test={data_cfg.get('test_ratio', 0.1):.0%}")
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
    train_counts  = train_labels.sum(axis=0)
    very_rare_div = float(train_cfg.get("very_rare_divisor", 3.0))
    rare_div      = float(train_cfg.get("rare_divisor",      1.0))
    very_rare_idx, rare_idx, common_idx = compute_tiers(train_counts, very_rare_div, rare_div)
    tier_indices = {"very_rare": very_rare_idx, "rare": rare_idx, "common": common_idx}

    print(f"[DataLoader] Stage={stage}  backbone={model_name}  num_workers={num_workers}")
    print(f"[DataLoader] Tier breakdown (train counts):")
    for i, name in enumerate(EMOTION_NAMES):
        tier = ("very_rare" if i in very_rare_idx else
                "rare"      if i in rare_idx else "common")
        print(f"    {name:<12}: {int(train_counts[i]):>6}  [{tier}]")

    # ── Stage-specific dataset config ─────────────────────────────────────────
    emotion_only_train = (stage == "stage2")
    augment_rare       = bool(train_cfg.get("augment_rare", False))
    aug_copies = {
        "very_rare": int(train_cfg.get("aug_copies_very_rare", 4)),
        "rare":      int(train_cfg.get("aug_copies_rare",      2)),
        "common":    int(train_cfg.get("aug_copies_common",    0)),
    }

    # [MOI] Stage 1 resampling config
    #   downsample_majority: true  +  downsample_ratio: 1.0
    #   oversample_minority: true  +  oversample_ratio: 1.0
    #   (co the bat dong thoi de ket hop, but thong thuong chon 1)
    resample_mode  = ""
    resample_ratio = 1.0
    if stage == "stage1":
        if bool(train_cfg.get("downsample_majority", False)):
            resample_mode  = "downsample"
            resample_ratio = float(train_cfg.get("downsample_ratio", 1.0))
        elif bool(train_cfg.get("oversample_minority", False)):
            resample_mode  = "oversample"
            resample_ratio = float(train_cfg.get("oversample_ratio", 1.0))

    # ── Build datasets ────────────────────────────────────────────────────────
    train_ds = EkmanDataset(
        train_texts, train_labels, tokenizer, max_length,
        stage=stage, emotion_only=emotion_only_train,
        augment_rare=augment_rare, aug_copies_per_tier=aug_copies,
        tier_indices=tier_indices,
        resample_mode=resample_mode,   # [MOI]
        resample_ratio=resample_ratio, # [MOI]
        resample_seed=seed,            # [MOI]
    )
    val_ds = EkmanDataset(
        val_texts, val_labels, tokenizer, max_length,
        stage=stage, emotion_only=emotion_only_train,
        # val/test: khong resample, danh gia tren phan phoi that
    )
    test_ds = EkmanDataset(
        test_texts, test_labels, tokenizer, max_length,
        stage=stage, emotion_only=emotion_only_train,
    )

    # ── Sampler ───────────────────────────────────────────────────────────────
    # Stage 1: use_weighted_sampler co the ket hop voi oversample de boost them
    # Stage 2: three-tier sampler cho rare/very_rare classes
    sampler     = None
    use_sampler = bool(train_cfg.get("use_weighted_sampler", False))

    if use_sampler and stage == "stage1":
        # [MOI] Binary sampler: weight ngach nho hon cao hon
        n_pos = float(train_ds.has_emotion.sum())
        n_neg = float(len(train_ds) - n_pos)
        w_pos = 1.0 / max(n_pos, 1.0)
        w_neg = 1.0 / max(n_neg, 1.0)
        sample_weights = np.where(train_ds.has_emotion > 0, w_pos, w_neg)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(train_ds),
            replacement=True,
        )
        print(f"[DataLoader] Stage1 WeightedSampler: "
              f"w_emotion={w_pos:.2e}  w_neutral={w_neg:.2e}  (effective ~1:1)")

    elif use_sampler and stage == "stage2":
        sampler = build_weighted_sampler(
            train_ds,
            sampler_power=   float(train_cfg.get("sampler_power",   2.0)),
            boost_very_rare= float(train_cfg.get("boost_very_rare", 5.0)),
            boost_rare=      float(train_cfg.get("boost_rare",      3.0)),
            boost_common=    float(train_cfg.get("boost_common",    1.0)),
            tier_indices=tier_indices,
        )

    train_loader = _make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, sampler=sampler)
    val_loader   = _make_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = _make_loader(test_ds,  batch_size, shuffle=False, num_workers=num_workers)

    # ── pos_weight ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if stage == "stage1":
        pos_weight = compute_pos_weight_stage1(
            train_ds, device,
            scale=float(train_cfg.get("pw_scale", 1.0)),
        )
        num_labels = 1
        print(f"[DataLoader] pos_weight (stage1): {pos_weight.item():.2f}")
    else:
        pos_weight = compute_pos_weight_stage2(
            train_ds, device,
            pw_scale_very_rare=float(train_cfg.get("pw_scale_very_rare", 3.0)),
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
    tr_t, tr_l = _load_csv(train_path)
    va_t, va_l = _load_csv(val_path)
    te_t, te_l = _load_csv(test_path)
    return tr_t, tr_l, va_t, va_l, te_t, te_l
