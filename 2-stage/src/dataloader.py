"""
src/dataloader.py
Two-Stage Ekman Emotion Classification — DataLoader.

Data format (CSV):
    text | anger | disgust | fear | joy | sadness | surprise

Stage modes:
    stage="stage1"  → binary label [has_emotion] — all samples
    stage="stage2"  → 6-hot label  [emotions]    — emotion-only samples

Imbalance handling:
    Stage 1 (2.5:1 emotion vs neutral):
        Majority (emotion) downsampling to a 1:1 ratio before training.
        Inverse-frequency pos_weight in BCEWithLogitsLoss.

    Stage 2 (6-class, up to 10× imbalance):
        1. Median-based synonym augmentation via nlpaug:
           classes below the median count are oversampled with synonym-replaced
           copies until they reach the median.
        2. Inverse-frequency pos_weight: w_c = n_neg_c / n_pos_c
           computed on the post-augmentation training distribution.
        3. Standard ASL (same γ⁺/γ⁻ for all classes): suppresses easy-negative
           dominance — orthogonal to class-frequency imbalance.

Inference:
    Sentence-level splitting + context window (mirrors inference.py).
    See sentence_predict() below.
"""

from __future__ import annotations

import os
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

EMOTION_NAMES:   List[str] = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
NUM_EMOTIONS:    int        = len(EMOTION_NAMES)   # 6
ALL_CLASS_NAMES: List[str] = EMOTION_NAMES + ["neutral"]
NUM_ALL_CLASSES: int        = 7

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


# =============================================================================
#  Sentence splitting (mirrors inference.py)
# =============================================================================

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
    Split a sentence that exceeds max_tokens at comma/semicolon boundaries.
    Falls back to hard truncation if no split points exist.
    """
    ids = tokenizer.encode(sent, add_special_tokens=True)
    if len(ids) <= max_tokens:
        return [sent]

    parts = _COMMA_RE.split(sent)
    if len(parts) == 1:
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
    Split a document into sentences, each fitting within max_tokens tokens.

    Step 1 — Sentence splitting via NLTK (or regex fallback).
    Step 2 — Long sentences are further split at comma/semicolon boundaries.

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

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    raw_sentences: List[str] = []
    for line in lines:
        parts = _nltk_split(line, language="english") if _USE_NLTK \
                else _SENT_RE.split(line)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if raw_sentences and len(raw_sentences[-1]) < 4:
                raw_sentences[-1] += " " + p
            else:
                raw_sentences.append(p)

    if not raw_sentences:
        return [text]

    result: List[str] = []
    for sent in raw_sentences:
        if tokenizer is not None:
            result.extend(_chunk_long_sentence(sent, tokenizer, max_tokens))
        else:
            est_tokens = len(sent.split()) * 1.3
            if est_tokens <= max_tokens:
                result.append(sent)
            else:
                parts = _COMMA_RE.split(sent)
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
    labels:     np.ndarray,
    tokenizer:  PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    device:     torch.device,
    window:     int = 0,
    stage:      str = "stage2",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference using sentence-level splitting with optional context window.

    For each document:
      1. Split into sentences (each fits within max_length tokens).
      2. For each sentence at index i, build the context string from
         up to `window` neighbouring sentences on each side.
      3. Run a forward pass per context string.
      4. Aggregate per-class probabilities across all sentences by
         element-wise maximum — a class is active in the document if
         any sentence triggers it above the threshold.

    Args:
        model:      Trained model in eval mode.
        texts:      List of preprocessed text strings.
        labels:     Ground-truth label matrix (returned unchanged).
        tokenizer:  Backbone tokenizer.
        max_length: Maximum sequence length used during training.
        batch_size: Forward-pass batch size.
        device:     Compute device.
        window:     Number of context sentences on each side (default 0).
        stage:      "stage1" (binary) or "stage2" (6-class).

    Returns:
        (probs, labels) where probs is (N, C) float32.
    """
    n_classes = 1 if stage == "stage1" else NUM_EMOTIONS
    model.eval()
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for text in texts:
            sentences = split_sentences(text, tokenizer=tokenizer,
                                        max_tokens=max_length)
            if not sentences:
                all_probs.append(np.zeros(n_classes, dtype=np.float32))
                continue

            ctx_strings: List[str] = []
            for idx in range(len(sentences)):
                if window == 0:
                    ctx = sentences[idx]
                else:
                    left  = sentences[max(0, idx - window): idx]
                    right = sentences[idx + 1: idx + 1 + window]
                    ctx   = " ".join(left + [sentences[idx]] + right)
                ctx_strings.append(ctx)

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
                probs = torch.sigmoid(logits).float().cpu().numpy()  # (B, C)
                doc_probs.extend(probs)

            # Aggregate: element-wise max across all sentence predictions
            doc_prob_max = np.max(doc_probs, axis=0)  # (C,)
            all_probs.append(doc_prob_max)

    return np.vstack(all_probs).astype(np.float32), labels


#  Stage 1 resampling

def _resample_stage1(
    texts:       List[str],
    labels_6:    np.ndarray,
    has_emotion: np.ndarray,
    mode:        str,
    ratio:       float,
    seed:        int = 42,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Resample Stage 1 training data to balance has_emotion vs neutral.

    Args:
        mode : "downsample" — remove majority samples to reach the target ratio.
               "oversample" — duplicate minority samples to reach the target ratio.
        ratio: target majority / minority ratio.
               1.0 → 1:1 balanced | 2.0 → 2:1 | 0.0 → disabled.

    Returns:
        (texts, labels_6, has_emotion) after resampling.
    """
    if ratio <= 0.0:
        return texts, labels_6, has_emotion

    texts     = list(texts)
    rng       = np.random.default_rng(seed)

    emotion_idx = np.where(has_emotion > 0)[0]
    neutral_idx = np.where(has_emotion == 0)[0]
    n_emotion   = len(emotion_idx)
    n_neutral   = len(neutral_idx)

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
        texts_out    = [texts[i] for i in final_idx]
        labels_out   = labels_6[final_idx]
        has_em_out   = has_emotion[final_idx]

    elif mode == "oversample":
        target_min   = min(int(n_maj * ratio), n_maj * 10)
        extra_needed = max(0, target_min - n_min)
        if extra_needed == 0:
            return texts, labels_6, has_emotion
        dup_idx      = rng.choice(minority_idx, size=extra_needed, replace=True)
        texts_out    = texts + [texts[i] for i in dup_idx]
        labels_out   = np.vstack([labels_6,    labels_6[dup_idx]])
        has_em_out   = np.concatenate([has_emotion, has_emotion[dup_idx]])

    else:
        raise ValueError(f"mode must be 'downsample' or 'oversample', got '{mode}'")

    n_new_em  = int(has_em_out.sum())
    n_new_neu = len(has_em_out) - n_new_em
    print(f"[DataLoader] stage1 {mode} (ratio={ratio}, majority={majority_label}): "
          f"{len(has_emotion)} → {len(has_em_out)} samples "
          f"(emotion={n_new_em}, neutral={n_new_neu})")
    return texts_out, labels_out, has_em_out


#  Median-based synonym augmentation for Stage 2 (nlpaug)

def _build_augmenter() -> naw.SynonymAug:
    """Build a WordNet synonym augmenter (called once per training run)."""
    return naw.SynonymAug(aug_src="wordnet", aug_p=0.15)


def augment_to_median_stage2(
    texts:    List[str],
    labels_6: np.ndarray,
    aug,
    seed:     int = 42,
) -> Tuple[List[str], np.ndarray]:
    """
    Oversample Stage 2 emotion classes below the median count using
    synonym augmentation (nlpaug SynonymAug, WordNet backend).

    For each class c with count n_c < median m, synthetic copies are
    generated by augmenting existing class-c samples until n_c reaches m.
    Classes at or above the median are left unchanged.

    Applied to training data only; val/test sets are never augmented.

    Args:
        texts:    List of preprocessed text strings.
        labels_6: (N, 6) float32 label matrix (emotion-only samples).
        aug:      Initialised nlpaug SynonymAug augmenter.
        seed:     Random seed for reproducibility.

    Returns:
        (augmented_texts, augmented_labels_6).
    """
    import random
    rng    = random.Random(seed)
    counts = labels_6.sum(axis=0)   # (6,) per-class positive counts
    median = float(np.median(counts))

    extra_texts:  List[str]        = []
    extra_labels: List[np.ndarray] = []

    for c in range(NUM_EMOTIONS):
        n_c = int(counts[c])
        if n_c >= median:
            continue

        pos_idx = np.where(labels_6[:, c] > 0)[0].tolist()
        if not pos_idx:
            continue

        needed  = int(median) - n_c
        pos_cycle = (pos_idx * (int(np.ceil(needed / len(pos_idx))) + 1))
        rng.shuffle(pos_cycle)
        generated = 0

        for orig_idx in pos_cycle:
            if generated >= needed:
                break
            try:
                result   = aug.augment(texts[orig_idx], n=1)
                aug_text = result[0] if isinstance(result, list) else result
            except Exception:
                aug_text = texts[orig_idx]   # fallback to original if augmenter fails
            extra_texts.append(aug_text)
            extra_labels.append(labels_6[orig_idx].copy())
            generated += 1

        print(f"[Augmentation] class={EMOTION_NAMES[c]:<12} "
              f"n_c={n_c:>6} → +{generated} samples (target median={int(median)})")

    if extra_texts:
        texts    = list(texts) + extra_texts
        labels_6 = np.vstack([labels_6, np.array(extra_labels)]).astype(np.float32)

    return texts, labels_6


#  Dataset
class EkmanDataset(Dataset):
    """
    Dataset for two-stage Ekman classification.

    Labels:
        stage1 → (1,) float32 binary [has_emotion]
        stage2 → (6,) float32 multi-label [emotion flags]

    Augmentation (median-based, Stage 2 only) is applied externally before
    constructing this dataset. The dataset itself only tokenises.
    """

    def __init__(
        self,
        texts:        List[str],
        labels_6:     np.ndarray,
        tokenizer,
        max_length:   int  = 128,
        stage:        str  = "stage2",
        emotion_only: bool = False,
    ) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.stage      = stage

        has_emotion = (labels_6.sum(axis=1) > 0).astype(np.float32)

        # Filter to emotion-positive samples for Stage 2 training
        if emotion_only:
            mask        = has_emotion.astype(bool)
            texts       = [t for t, m in zip(texts, mask) if m]
            labels_6    = labels_6[mask]
            has_emotion = has_emotion[mask]

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
            item["labels"] = torch.tensor(self.labels_6[idx], dtype=torch.float32)
        return item


#  Positive-weight computation

def compute_pos_weight_stage1(
    has_emotion: np.ndarray,
    device:      torch.device,
) -> torch.Tensor:
    """
    (1,) pos_weight for Stage 1 binary BCE.
    w = n_neutral / n_emotion (inverse frequency).
    """
    n_emotion = max(float(has_emotion.sum()), 1.0)
    n_neutral = max(float(len(has_emotion) - n_emotion), 1.0)
    pw = float(np.clip(n_neutral / n_emotion, 0.01, 200.0))
    print(f"[DataLoader] pos_weight (stage1): {pw:.2f}")
    return torch.tensor([pw], dtype=torch.float32, device=device)


def compute_pos_weight_stage2(
    labels_6: np.ndarray,
    device:   torch.device,
) -> torch.Tensor:
    """
    (6,) per-class pos_weight for Stage 2 ASL.

    w_c = n_neg_c / n_pos_c

    Computed on the post-augmentation label matrix so the weights reflect
    the balanced distribution after augmentation.
    """
    n          = labels_6.shape[0]
    pos_counts = np.maximum((labels_6 > 0).sum(axis=0).astype(np.float64), 1.0)
    neg_counts = np.maximum(n - pos_counts, 1.0)
    pw         = np.clip(neg_counts / pos_counts, 0.01, 200.0).astype(np.float32)

    print("[DataLoader] pos_weight (stage2, post-augmentation):")
    for i, name in enumerate(EMOTION_NAMES):
        print(f"    {name:<12}: {pw[i]:.2f}")

    return torch.tensor(pw, dtype=torch.float32, device=device)


#  CSV loading & splitting

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


def _split_data(
    texts:      List[str],
    labels_6:   np.ndarray,
    val_ratio:  float = 0.10,
    test_ratio: float = 0.10,
    seed:       int   = 42,
) -> Tuple:
    """Stratified two-stage split preserving emotion/neutral balance."""
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


def _make_loader(
    dataset:     EkmanDataset,
    batch_size:  int,
    shuffle:     bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


#  Main factory

def get_dataloaders(
    cfg:   dict,
    stage: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Build train/val/test DataLoaders for the given stage.

    Augmentation (median-based, nlpaug) is applied to Stage 2 training data
    only. Stage 1 training data is downsampled to 1:1 ratio. Val and test
    datasets are always unmodified. Sentence-level inference is handled
    separately by sentence_predict().

    Returns:
        (train_loader, val_loader, test_loader, info_dict)

    info_dict keys:
        emotion_names, pos_weight, label_counts, num_labels, pretrained,
        tokenizer, max_length, batch_size,
        val_texts, val_labels, test_texts, test_labels
    """
    assert stage in ("stage1", "stage2"), f"Unknown stage '{stage}'"

    data_cfg  = cfg["data"]
    stage_cfg = cfg[stage]
    train_cfg = stage_cfg["training"]

    max_length  = int(data_cfg.get("max_length",  128))
    batch_size  = int(train_cfg.get("batch_size", 32))
    seed        = int(data_cfg.get("seed",        42))
    auto_split  = bool(data_cfg.get("auto_split", False))
    num_workers = int(data_cfg.get("num_workers", 2))

    model_name = stage_cfg["model"]["name"].lower()
    if model_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {' | '.join(BACKBONE_REGISTRY.keys())}"
        )
    pretrained = BACKBONE_REGISTRY[model_name]["pretrained"]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)

    # Load raw data 
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

    print(f"[DataLoader] stage={stage}  backbone={model_name}  num_workers={num_workers}")
    has_emotion_train = (train_labels_6.sum(axis=1) > 0).astype(np.float32)

    if stage == "stage1":
        # ── Stage 1:───────────────────────
        sample_mode = train_cfg.get("sample", "down")  
        ratio       = float(train_cfg.get("ratio", 1.0))

        # map sang mode đầy đủ
        if sample_mode == "down":
            train_texts, train_labels_6, has_emotion_train = _resample_stage1(
                train_texts,
                train_labels_6,
                has_emotion_train,
                mode="downsample",
                ratio=ratio,
                seed=seed,
            )
            
        elif sample_mode == "over":
            train_texts, train_labels_6, has_emotion_train = _resample_stage1(
                train_texts,
                train_labels_6,
                has_emotion_train,
                mode="oversample",
                ratio=ratio,
                seed=seed,
            )

        train_ds = EkmanDataset(train_texts, train_labels_6, tokenizer, max_length,
                                stage="stage1", emotion_only=False)
        val_ds   = EkmanDataset(val_texts,   val_labels_6,   tokenizer, max_length,
                                stage="stage1", emotion_only=False)
        test_ds  = EkmanDataset(test_texts,  test_labels_6,  tokenizer, max_length,
                                stage="stage1", emotion_only=False)

        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos_weight = compute_pos_weight_stage1(train_ds.has_emotion, device)
        num_labels = 1

        # Val/test labels for sentence_predict (binary: 1 = has emotion)
        val_labels_out  = (val_labels_6.sum(axis=1,  keepdims=True) > 0).astype(np.float32)
        test_labels_out = (test_labels_6.sum(axis=1, keepdims=True) > 0).astype(np.float32)

    else:
        # ── Stage 2: filter emotion-only, then median augmentation ────────────
        # Filter train to emotion-only first (before augmentation)
        emo_mask         = has_emotion_train.astype(bool)
        train_texts_emo  = [t for t, m in zip(train_texts, emo_mask) if m]
        train_labels_emo = train_labels_6[emo_mask]

        do_augment = bool(train_cfg.get("augment", True))
        if do_augment:
            print("[DataLoader] Running median-based synonym augmentation (nlpaug)...")
            aug = _build_augmenter()
            train_texts_emo, train_labels_emo = augment_to_median_stage2(
                train_texts_emo, train_labels_emo, aug, seed=seed
            )
            print(f"[DataLoader] Stage 2 training set after augmentation: "
                  f"{len(train_texts_emo)} samples\n")
        else:
            print("[DataLoader] Augmentation disabled (stage2).")

        train_ds = EkmanDataset(train_texts_emo, train_labels_emo, tokenizer, max_length,
                                stage="stage2", emotion_only=False)
        # Val/test: emotion-only subset (matches stage2 training distribution)
        val_emo_mask  = (val_labels_6.sum(axis=1)  > 0)
        test_emo_mask = (test_labels_6.sum(axis=1) > 0)
        val_ds  = EkmanDataset(
            [t for t, m in zip(val_texts,  val_emo_mask)  if m],
            val_labels_6[val_emo_mask],  tokenizer, max_length, stage="stage2",
        )
        test_ds = EkmanDataset(
            [t for t, m in zip(test_texts, test_emo_mask) if m],
            test_labels_6[test_emo_mask], tokenizer, max_length, stage="stage2",
        )

        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos_weight = compute_pos_weight_stage2(train_labels_emo, device)
        num_labels = NUM_EMOTIONS

        val_labels_out  = val_ds.labels_6
        test_labels_out = test_ds.labels_6

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = _make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = _make_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = _make_loader(test_ds,  batch_size, shuffle=False, num_workers=num_workers)

    label_counts_dict = {EMOTION_NAMES[i]: int(train_ds.labels_6[:, i].sum())
                         for i in range(NUM_EMOTIONS)}
    ne = int(train_ds.has_emotion.sum())
    label_counts_dict.update({"has_emotion": ne, "neutral": len(train_ds) - ne})

    print(f"[DataLoader] Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")

    return train_loader, val_loader, test_loader, {
        "emotion_names":  EMOTION_NAMES,
        "pos_weight":     pos_weight,
        "label_counts":   label_counts_dict,
        "num_labels":     num_labels,
        "pretrained":     pretrained,
        "tokenizer":      tokenizer,
        "max_length":     max_length,
        "batch_size":     batch_size,
        # Raw text + labels for sentence_predict
        "val_texts":      val_ds.texts,
        "val_labels":     val_labels_out,
        "test_texts":     test_ds.texts,
        "test_labels":    test_labels_out,
    }


def get_raw_splits(cfg: dict) -> Tuple:
    """Return raw (train_texts, train_labels_6, ...) splits without any processing."""
    data_cfg   = cfg["data"]
    data_dir   = data_cfg["data_dir"]
    auto_split = bool(data_cfg.get("auto_split", False))
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
