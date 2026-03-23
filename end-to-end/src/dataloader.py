"""
src/dataloader.py
Single-Stage End-to-End Ekman Classification — DataLoader.

Task: 7-class multi-label (6 emotions + neutral) on ALL samples.

Data format (CSV):
    text | anger | disgust | fear | joy | sadness | surprise
    - Emotion samples : one or more emotion columns = 1
    - Neutral samples : all 6 emotion columns = 0  →  neutral label auto-derived

Imbalance handling:
    1. Median-based synonym augmentation via nlpaug (training only):
       Classes whose count falls below the median are oversampled with
       synonym-replaced copies until they reach the median count.
    2. Inverse-frequency pos_weight in the loss (w_c = n_neg_c / n_pos_c).

Inference (val / test):
    Sliding-window aggregation for texts longer than max_length tokens.
    See sliding_window_predict() below.
"""

from __future__ import annotations

import os
import random
import re
from typing import Dict, List, Optional, Tuple

import nlpaug.augmenter.word as naw
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


#  Backbone registry

BACKBONE_REGISTRY: Dict[str, Dict[str, str]] = {
    "bert":    {"pretrained": "google-bert/bert-base-uncased",     "amp_dtype": "float16"},
    "roberta": {"pretrained": "FacebookAI/roberta-base",           "amp_dtype": "float16"},
    "electra": {"pretrained": "google/electra-base-discriminator", "amp_dtype": "float16"},
}


#  Label metadata
EMOTION_NAMES: List[str] = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
NUM_EMOTIONS:  int        = len(EMOTION_NAMES)       # 6
CLASS_NAMES:   List[str]  = EMOTION_NAMES + ["neutral"]
NUM_CLASSES:   int        = len(CLASS_NAMES)          # 7
NEUTRAL_IDX:   int        = 6


#  Text preprocessing

_RE_URL     = re.compile(r"https?://\S+|www\.\S+")
_RE_MENTION = re.compile(r"@\w+")
_RE_SPACES  = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    """
    Clean raw social-media text before tokenisation.

    Steps (order matters):
      1. Replace URLs     → '<url>'
      2. Replace mentions → '<mention>'
      3. Collapse extra whitespace → single space
    """
    text = _RE_URL.sub("<url>", text)
    text = _RE_MENTION.sub("<mention>", text)
    text = _RE_SPACES.sub(" ", text).strip()
    return text


#  Median-based synonym augmentation (nlpaug)

def _build_augmenter() -> naw.SynonymAug:
    """Build a WordNet synonym augmenter (lazy init, called once)."""
    return naw.SynonymAug(aug_src="wordnet", aug_p=0.15)


def augment_to_median(
    texts:    List[str],
    labels_7: np.ndarray,
    aug:      naw.SynonymAug,
    seed:     int = 42,
) -> Tuple[List[str], np.ndarray]:
    """
    Oversample classes below the median count using synonym augmentation.

    For each class c whose count n_c < median m, synthetic copies of
    class-c samples are generated until the count reaches m.
    The number of copies per original sample is ceil((m - n_c) / n_c).
    Classes at or above the median are left unchanged.

    Args:
        texts:    List of preprocessed text strings (training set only).
        labels_7: (N, 7) float32 label matrix.
        aug:      Initialised nlpaug SynonymAug augmenter.
        seed:     Random seed for reproducibility.

    Returns:
        (augmented_texts, augmented_labels_7) with augmented samples appended.
    """
    rng = random.Random(seed)
    counts = labels_7.sum(axis=0)          # (7,) — positive count per class
    median = float(np.median(counts))

    extra_texts:  List[str]        = []
    extra_labels: List[np.ndarray] = []

    for c in range(NUM_CLASSES):
        n_c = int(counts[c])
        if n_c >= median:
            continue  # already at or above median

        # Indices of samples that are positive for class c
        pos_idx = np.where(labels_7[:, c] > 0)[0].tolist()
        if not pos_idx:
            continue

        copies_per_sample = int(np.ceil((median - n_c) / n_c))
        needed = int(median) - n_c
        generated = 0

        # Cycle through positive samples until deficit is filled
        pos_cycle = pos_idx * (copies_per_sample + 1)
        rng.shuffle(pos_cycle)

        for orig_idx in pos_cycle:
            if generated >= needed:
                break
            orig_text = texts[orig_idx]
            aug_seed  = rng.randint(0, 10_000_000)
            try:
                aug_result = aug.augment(orig_text, n=1)
                aug_text   = aug_result[0] if isinstance(aug_result, list) else aug_result
            except Exception:
                aug_text = orig_text   # fallback: keep original if augmenter fails
            extra_texts.append(aug_text)
            extra_labels.append(labels_7[orig_idx].copy())
            generated += 1

        print(f"[Augmentation] class={CLASS_NAMES[c]:<12} "
              f"n_c={n_c:>6} → +{generated} samples (target median={int(median)})")

    if extra_texts:
        aug_labels = np.vstack(extra_labels).astype(np.float32)
        texts      = list(texts) + extra_texts
        labels_7   = np.vstack([labels_7, aug_labels])

    return texts, labels_7


#  Dataset

class EkmanDataset(Dataset):
    """
    7-class multi-label dataset for end-to-end Ekman classification.

    Labels: (7,) float32
        cols 0-5 = emotion flags  (EMOTION_NAMES order)
        col  6   = neutral         (1 iff all emotion cols = 0)

    Note: Augmentation is applied externally before constructing this dataset.
          The dataset itself only handles tokenisation (with truncation).
          For sliding-window inference on long texts use sliding_window_predict().
    """

    def __init__(
        self,
        texts:      List[str],
        labels_7:   np.ndarray,
        tokenizer:  PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.texts      = list(texts)
        self.labels_7   = labels_7.astype(np.float32)

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



#  Sentence splitting (mirrors inference.py)

try:
    import nltk
    for _pkg in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{_pkg}")
        except LookupError:
            nltk.download(_pkg, quiet=True)
    from nltk.tokenize import sent_tokenize as _nltk_split
    _USE_NLTK = True
except ImportError:
    _USE_NLTK = False

_SENT_RE  = re.compile(r'(?<=[.!?…])\s+', re.UNICODE)
_COMMA_RE = re.compile(r'(?<=,|;)\s+')


def _chunk_long_sentence(
    sent:       str,
    tokenizer:  PreTrainedTokenizerBase,
    max_tokens: int,
) -> List[str]:
    """
    Split a sentence that exceeds max_tokens at comma / semicolon boundaries.
    Falls back to hard truncation if no split points exist.
    """
    ids = tokenizer.encode(sent, add_special_tokens=True)
    if len(ids) <= max_tokens:
        return [sent]

    parts = _COMMA_RE.split(sent)
    if len(parts) == 1:
        # No comma found — hard truncate (rare)
        safe_ids = ids[:max_tokens - 1]
        return [tokenizer.decode(safe_ids, skip_special_tokens=True)]

    chunks: List[str] = []
    current = ""
    for part in parts:
        candidate = (current + ", " + part).strip(", ") if current else part
        if len(tokenizer.encode(candidate, add_special_tokens=True)) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks or [sent]


def split_sentences(
    text:       str,
    tokenizer:  Optional[PreTrainedTokenizerBase] = None,
    max_tokens: int = 128,
) -> List[str]:
    """
    Split a document into sentences, ensuring each sentence fits within
    max_tokens tokens.

    Step 1 — Sentence splitting via NLTK (or regex fallback).
    Step 2 — Long sentences are further split at comma / semicolon boundaries.

    Args:
        text:       Input text (pre-cleaned).
        tokenizer:  Backbone tokenizer for exact token counting.
                    If None, word count × 1.3 is used as an estimate.
        max_tokens: Maximum tokens per chunk (should equal config max_length).

    Returns:
        List of sentence strings, each guaranteed to fit within max_tokens.
    """
    text = text.strip()
    if not text:
        return []

    # Step 1: sentence splitting
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    raw_sentences: List[str] = []
    for line in lines:
        parts = _nltk_split(line, language="english") if _USE_NLTK \
                else _SENT_RE.split(line)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # Merge orphan fragments (< 4 chars) with the previous sentence
            if raw_sentences and len(raw_sentences[-1]) < 4:
                raw_sentences[-1] += " " + p
            else:
                raw_sentences.append(p)

    if not raw_sentences:
        return [text]

    # Step 2: chunk long sentences
    result: List[str] = []
    for sent in raw_sentences:
        if tokenizer is not None:
            result.extend(_chunk_long_sentence(sent, tokenizer, max_tokens))
        else:
            est_tokens = len(sent.split()) * 1.3
            if est_tokens <= max_tokens:
                result.append(sent)
            else:
                parts  = _COMMA_RE.split(sent)
                chunk, chunk_words = "", 0
                max_words = int(max_tokens / 1.3)
                for part in parts:
                    w = len(part.split())
                    if chunk_words + w <= max_words:
                        chunk = (chunk + ", " + part).strip(", ") if chunk else part
                        chunk_words += w
                    else:
                        if chunk:
                            result.append(chunk)
                        chunk, chunk_words = part, w
                if chunk:
                    result.append(chunk)
    return result or [text]


#  Sentence-level inference helper

def sentence_predict(
    model:      torch.nn.Module,
    texts:      List[str],
    labels_7:   np.ndarray,
    tokenizer:  PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    device:     torch.device,
    window:     int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference using sentence-level splitting with optional context window.

    For each document:
      1. Split into sentences (each fits within max_length tokens).
      2. For each sentence at index i, build the context string by concatenating
         up to `window` sentences on each side:
             ctx = sentences[i-window : i+window+1]
      3. Run a single forward pass for each context string.
      4. Aggregate per-class probabilities across all sentences by taking the
         element-wise maximum — a class is considered active in the document
         if any sentence triggers it above the threshold.

    With window=0 each sentence is predicted independently; with window=1
    one neighbouring sentence on each side is included as context.

    Args:
        model:      Trained model in eval mode.
        texts:      List of preprocessed document strings.
        labels_7:   (N, 7) ground-truth label matrix (returned unchanged).
        tokenizer:  Backbone tokenizer.
        max_length: Maximum sequence length used during training.
        batch_size: Forward-pass batch size (context strings are batched).
        device:     Compute device.
        window:     Number of context sentences on each side (default 0).

    Returns:
        (probs, labels_7) where probs is (N, 7) float32.
    """
    model.eval()
    all_probs: List[np.ndarray] = []

    # Collect all context strings per document, then batch across documents
    with torch.no_grad():
        for text in texts:
            sentences = split_sentences(text, tokenizer=tokenizer,
                                        max_tokens=max_length)
            if not sentences:
                # Empty text: predict zeros
                all_probs.append(np.zeros(NUM_CLASSES, dtype=np.float32))
                continue

            # Build context strings for each sentence position
            ctx_strings: List[str] = []
            for idx in range(len(sentences)):
                if window == 0:
                    ctx = sentences[idx]
                else:
                    left  = sentences[max(0, idx - window): idx]
                    right = sentences[idx + 1: idx + 1 + window]
                    ctx   = " ".join(left + [sentences[idx]] + right)
                ctx_strings.append(ctx)

            # Forward pass in mini-batches
            doc_probs: List[np.ndarray] = []
            for b_start in range(0, len(ctx_strings), batch_size):
                batch_ctx = ctx_strings[b_start: b_start + batch_size]
                enc = tokenizer(
                    batch_ctx,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                logits = model(
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device),
                )
                probs = torch.sigmoid(logits).float().cpu().numpy()  # (B, 7)
                doc_probs.extend(probs)

            # Aggregate: take element-wise max across all sentences
            doc_prob_max = np.max(doc_probs, axis=0)  # (7,)
            all_probs.append(doc_prob_max)

    return np.vstack(all_probs).astype(np.float32), labels_7



#  Positive-weight computation

def compute_pos_weight(
    labels_7: np.ndarray,
    device:   torch.device,
) -> torch.Tensor:
    """
    Compute per-class inverse-frequency positive weights for BCEWithLogitsLoss.

    w_c = n_neg_c / n_pos_c

    Computed on the post-augmentation training label matrix so that the
    weights reflect the balanced distribution after augmentation.
    """
    n          = labels_7.shape[0]
    pos_counts = np.maximum((labels_7 > 0).sum(axis=0).astype(np.float64), 1.0)
    neg_counts = np.maximum(n - pos_counts, 1.0)
    pw         = (neg_counts / pos_counts).astype(np.float32)

    print("[DataLoader] pos_weight (post-augmentation):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:<12}: {pw[i]:.2f}")

    return torch.tensor(pw, dtype=torch.float32, device=device)


# =============================================================================
#  CSV loading & auto-splitting
# =============================================================================

def _load_csv(filepath: str) -> Tuple[List[str], np.ndarray]:
    """Load CSV and return (texts, labels_6) where labels_6 is (N, 6) float32."""
    df      = pd.read_csv(filepath)
    missing = [c for c in EMOTION_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {filepath}")
    if "text" not in df.columns:
        raise ValueError(f"'text' column not found in {filepath}")
    texts = [preprocess_text(t) for t in df["text"].astype(str).tolist()]
    return texts, df[EMOTION_NAMES].values.astype(np.float32)


def _build_labels7(labels_6: np.ndarray) -> np.ndarray:
    """Append neutral column: neutral = 1 iff all 6 emotion flags are 0."""
    neutral_col = (labels_6.sum(axis=1) == 0).astype(np.float32).reshape(-1, 1)
    return np.concatenate([labels_6, neutral_col], axis=1).astype(np.float32)


def _split_data(
    texts:      List[str],
    labels_6:   np.ndarray,
    val_ratio:  float = 0.10,
    test_ratio: float = 0.10,
    seed:       int   = 42,
) -> Tuple:
    """Stratified two-stage split: test first, then val from the remainder."""
    strat = (labels_6.sum(axis=1) > 0).astype(int)
    tv_t, test_t, tv_l, test_l = train_test_split(
        texts, labels_6, test_size=test_ratio, random_state=seed, stratify=strat,
    )
    strat2 = (tv_l.sum(axis=1) > 0).astype(int)
    val_frac = val_ratio / (1.0 - test_ratio)
    tr_t, val_t, tr_l, val_l = train_test_split(
        tv_t, tv_l, test_size=val_frac, random_state=seed, stratify=strat2,
    )
    return tr_t, tr_l, val_t, val_l, test_t, test_l



#  Main factory

def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Build train / val / test DataLoaders for the 7-class E2E task.

    Augmentation (median-based, nlpaug) is applied to the training set only.
    Val and test datasets use simple truncation; sliding-window inference is
    handled separately in train.py (_run_epoch) and test.py (_infer).

    Returns:
        (train_loader, val_loader, test_loader, info_dict)

    info_dict keys:
        class_names, pos_weight, label_counts, num_labels, pretrained,
        tokenizer, max_length, batch_size
    """
    data_cfg  = cfg["data"]
    stage_cfg = cfg["e2e"]
    train_cfg = stage_cfg["training"]

    max_length  = int(data_cfg.get("max_length",  128))
    batch_size  = int(train_cfg.get("batch_size", 32))
    seed        = int(data_cfg.get("seed",        42))
    auto_split  = bool(data_cfg.get("auto_split", False))
    num_workers = int(data_cfg.get("num_workers", 2))
    do_augment  = bool(train_cfg.get("augment",   True))

    model_name = stage_cfg["model"]["name"].lower()
    if model_name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {' | '.join(BACKBONE_REGISTRY)}")
    pretrained = BACKBONE_REGISTRY[model_name]["pretrained"]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)

    # ── Load raw data ─────────────────────────────────────────────────────────
    data_dir   = data_cfg["data_dir"]
    train_path = os.path.join(data_dir, data_cfg.get("train_file", "data1_train.csv"))
    val_path   = os.path.join(data_dir, data_cfg.get("val_file",   "data1_val.csv"))
    test_path  = os.path.join(data_dir, data_cfg.get("test_file",  "data1_test.csv"))

    if auto_split and (not os.path.isfile(val_path) or not os.path.isfile(test_path)):
        print(f"[DataLoader] Auto-splitting '{train_path}'")
        all_t, all_l = _load_csv(train_path)
        train_texts, train_labels_6, val_texts, val_labels_6, test_texts, test_labels_6 = \
            _split_data(all_t, all_l,
                        val_ratio=float(data_cfg.get("val_ratio",  0.10)),
                        test_ratio=float(data_cfg.get("test_ratio", 0.10)),
                        seed=seed)
    else:
        for split, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"[DataLoader] {split} file not found: '{path}'")
        train_texts, train_labels_6 = _load_csv(train_path)
        val_texts,   val_labels_6   = _load_csv(val_path)
        test_texts,  test_labels_6  = _load_csv(test_path)

    # ── Build 7-dim label matrices ────────────────────────────────────────────
    train_labels_7 = _build_labels7(train_labels_6)
    val_labels_7   = _build_labels7(val_labels_6)
    test_labels_7  = _build_labels7(test_labels_6)

    # ── Median-based augmentation (training only) ─────────────────────────────
    if do_augment:
        print(f"\n[DataLoader] Running median-based synonym augmentation (nlpaug)...")
        aug = _build_augmenter()
        train_texts, train_labels_7 = augment_to_median(
            train_texts, train_labels_7, aug, seed=seed
        )
        print(f"[DataLoader] Training set after augmentation: {len(train_texts)} samples\n")
    else:
        print("[DataLoader] Augmentation disabled.")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = EkmanDataset(train_texts, train_labels_7, tokenizer, max_length)
    val_ds   = EkmanDataset(val_texts,   val_labels_7,   tokenizer, max_length)
    test_ds  = EkmanDataset(test_texts,  test_labels_7,  tokenizer, max_length)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # ── Pos-weight (computed from post-augmentation training labels) ──────────
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = compute_pos_weight(train_labels_7, device)

    label_counts_dict = {
        CLASS_NAMES[i]: int(train_labels_7[:, i].sum()) for i in range(NUM_CLASSES)
    }
    print(f"[DataLoader] backbone={model_name}  |  pretrained={pretrained}")
    print(f"[DataLoader] Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")

    return train_loader, val_loader, test_loader, {
        "class_names":  CLASS_NAMES,
        "pos_weight":   pos_weight,
        "label_counts": label_counts_dict,
        "num_labels":   NUM_CLASSES,
        "pretrained":   pretrained,
        "tokenizer":    tokenizer,
        "max_length":   max_length,
        "batch_size":   batch_size,
        # Raw splits kept for sliding-window inference
        "val_texts":    val_texts,
        "val_labels_7": val_labels_7,
        "test_texts":   test_texts,
        "test_labels_7":test_labels_7,
    }
