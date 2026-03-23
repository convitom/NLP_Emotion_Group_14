from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# resolve project root 
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataloader import BACKBONE_REGISTRY, CLASS_NAMES, NUM_CLASSES
from src.train import build_model
from src.utils import load_config, set_seed
from grad_cam import TokenSaliency, compute_saliency_for_classes

NEUTRAL_IDX   = CLASS_NAMES.index("neutral")
EMOTION_NAMES = [c for c in CLASS_NAMES if c != "neutral"]

#  CONFIG 

model = "electra"  
CHECKPOINT = r"D:\USTH\nlp\NLP_Emotion_Group_14\end-to-end\results\{model}\checkpoints\{model}_e2e_emotion.pth".format(model=model)

CONFIG = r"D:\USTH\nlp\NLP_Emotion_Group_14\end-to-end\config\config.yaml"

WINDOW    = 0    
TOP_K     = 10    
SALIENCY  = True   
SHOW_JSON = False  # True = in JSON 


TEXT_1 = """
Last night was fire!
"""


TEXTS = [TEXT_1]



#  Model 
 
class EncoderForClassification(nn.Module):
 
    def __init__(self, pretrained_name: str, num_labels: int = NUM_CLASSES,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone    = AutoModel.from_pretrained(pretrained_name)
        self.dropout     = nn.Dropout(dropout)
        hidden           = self.backbone.config.hidden_size
        self.classifiers = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(num_labels)]
        )
        self.num_labels  = num_labels
 
    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return torch.cat([h(cls) for h in self.classifiers], dim=1)  # (B, 7)
 
 
#  Output data classes
@dataclass
class EmotionScore:
    emotion:     str
    probability: float
    detected:    bool
 
 
@dataclass
class SentenceResult:
    sentence_idx:    int
    sentence_text:   str
    context_text:    str
    emotions:        List[EmotionScore]
    token_saliency:  Dict[str, List[TokenSaliency]]
    is_neutral:      bool
 
 
@dataclass
class DocumentResult:
    full_text:          str
    sentences:          List[SentenceResult]
    detected_emotions:  List[str]
    emotion_confidence: Dict[str, float]
    is_neutral:         bool
    top_tokens:         Dict[str, List[Tuple[str, float]]]
 
    def __str__(self) -> str:
        lines = ["=" * 64]
        preview = self.full_text.strip()[:110]
        if len(self.full_text.strip()) > 110:
            preview += "…"
        lines.append(f"  Input : {preview}")
        lines.append("=" * 64)
 
        if self.is_neutral:
            lines.append("  Result : NEUTRAL")
        else:
            lines.append("  Result : HAS EMOTION")
            for emo in self.detected_emotions:
                conf = self.emotion_confidence[emo]
                bar  = "█" * int(conf * 20)
                lines.append(f"    • {emo:<12}  {conf:.3f}  {bar}")
 
        if self.top_tokens:
            lines.append("")
            lines.append("  Influential tokens (Grad-CAM):")
            for emo, toks in self.top_tokens.items():
                tok_str = "  ".join(f'"{t}"({s:.2f})' for t, s in toks[:8])
                lines.append(f"    [{emo}]  {tok_str}")
 
        lines.append("")
        lines.append(f"  Sentences ({len(self.sentences)}):")
        for sr in self.sentences:
            emos = [e.emotion for e in sr.emotions
                    if e.detected and e.emotion != "neutral"]
            tag  = ", ".join(emos) if emos else "neutral"
            lines.append(f"    [{sr.sentence_idx}] {sr.sentence_text[:72]!r}")
            lines.append(f"          → {tag}")
 
        lines.append("=" * 64)
        return "\n".join(lines)
 
    def to_dict(self) -> dict:
        return {
            "full_text":          self.full_text,
            "is_neutral":         self.is_neutral,
            "detected_emotions":  self.detected_emotions,
            "emotion_confidence": self.emotion_confidence,
            "top_tokens":         {e: list(t) for e, t in self.top_tokens.items()},
            "sentences": [
                {
                    "idx":        sr.sentence_idx,
                    "text":       sr.sentence_text,
                    "context":    sr.context_text,
                    "is_neutral": sr.is_neutral,
                    "emotions": [
                        {"emotion":  e.emotion,
                         "prob":     round(e.probability, 4),
                         "detected": e.detected}
                        for e in sr.emotions
                    ],
                    "token_saliency": {
                        emo: [{"token": t.token, "score": round(t.score, 4)}
                              for t in sal]
                        for emo, sal in sr.token_saliency.items()
                    },
                }
                for sr in self.sentences
            ],
        }
 
#  Sentence splitter
 
# nltk sentence tokenizer 
try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    from nltk.tokenize import sent_tokenize as _nltk_split
    _USE_NLTK = True
except ImportError:
    _USE_NLTK = False
 
# Fallback regex — when nltk is not available
_SENT_RE = re.compile(r'(?<=[.!?…])\s+', re.UNICODE)
 
 
# Chunk long sentences at commas/semicolons if they exceed max_tokens
_COMMA_RE = re.compile(r'(?<=,|;)\s+')
 
 
def _chunk_long_sentence(
    sent: str, tokenizer, max_tokens: int
) -> List[str]:
    """
    If a sentence exceeds max_tokens, split it at the nearest comma or semicolon.
    Return a list of chunks, each with length <= max_tokens.
    If no commas or semicolons are found, truncate hard (rare case).
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
        n_tokens  = len(tokenizer.encode(candidate, add_special_tokens=True))
        if n_tokens <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks or [sent]
 
 
def split_sentences(
    text: str,
    tokenizer=None,
    max_tokens: int = 128,
) -> List[str]:
    """
    Split the text into a list of sentences, ensuring each sentence has <= max_tokens tokens.

    Step 1 — Split sentences using NLTK (or a regex fallback).
    Step 2 — If a sentence is still too long, further split it at commas / semicolons.

    Parameters
    ----------
    tokenizer : AutoTokenizer | None
        If provided, token counts are computed accurately.
        If None, estimate using words * 1.3 (safer).

    max_tokens : int
        Maximum token threshold per chunk (should match max_length in the config).
    """
    text = text.strip()
    if not text:
        return []
 
    # Step 1: split into raw sentences
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    raw_sentences: List[str] = []
    for line in lines:
        if _USE_NLTK:
            parts = _nltk_split(line, language="english")
        else:
            parts = _SENT_RE.split(line)
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
 
    # Step 2: chunk long sentences
    # Use tokenizer if available, fallback to word-based estimation
    result: List[str] = []
    for sent in raw_sentences:
        if tokenizer is not None:
            result.extend(_chunk_long_sentence(sent, tokenizer, max_tokens))
        else:
            # Estimate: 1 word ≈ 1.3 tokens (average BPE/WordPiece)
            est_tokens = len(sent.split()) * 1.3
            if est_tokens <= max_tokens:
                result.append(sent)
            else:
                # No actual tokenizer available → split by commas manually
                parts = _COMMA_RE.split(sent)
                chunk, chunk_words = "", 0
                for part in parts:
                    w = len(part.split())
                    if chunk_words + w <= int(max_tokens / 1.3):
                        chunk = (chunk + ", " + part).strip(", ") if chunk else part
                        chunk_words += w
                    else:
                        if chunk:
                            result.append(chunk)
                        chunk, chunk_words = part, w
                if chunk:
                    result.append(chunk)
    return result or [text]
 
#  Inference engine

class EmotionInference:
 
    def __init__(
        self,
        checkpoint_path:  str,
        config_path:      str,
        device:           Optional[str] = None,
        compute_saliency: bool = True,
    ):
        cfg = load_config(config_path)
        set_seed(cfg["data"]["seed"])
 
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"[inference] device={self.device}")
 
        self.compute_saliency = compute_saliency
        self.max_length       = int(cfg["data"].get("max_length", 128))
 
        # Load checkpoint 
        print(f"[inference] Loading: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
 
        model_name      = ckpt["model_name"]
        pretrained_name = ckpt["pretrained_name"]
        num_labels      = ckpt.get("num_labels", NUM_CLASSES)
        self.threshold  = float(ckpt.get("threshold", 0.5))
 
        print(f"  model={model_name}  pretrained={pretrained_name}")
        print(f"  epoch={ckpt.get('epoch', '?')}  "
              f"threshold={self.threshold}  num_labels={num_labels}")
 
        # Build model directly from pretrained_name in checkpoint 
        dropout = float(cfg["e2e"]["model"].get("dropout", 0.1))
        self.model = EncoderForClassification(
            pretrained_name=pretrained_name,
            num_labels=num_labels,
            dropout=dropout,
        ).to(self.device)
 
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
 
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        print("[inference] Ready.\n")
 
    #  Helpers
 
    def _enc(self, text: str) -> dict:
        return self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
 
    @torch.no_grad()
    def _forward_probs(self, text: str) -> np.ndarray:
        e = self._enc(text)
        logits = self.model(
            e["input_ids"].to(self.device),
            e["attention_mask"].to(self.device),
        )
        return torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (7,)
 
    def _saliency(
        self, text: str, class_indices: List[int]
    ) -> Dict[int, List[TokenSaliency]]:
        enc = self._enc(text)
        for p in self.model.parameters():
            p.requires_grad_(True)
        try:
            sal = compute_saliency_for_classes(
                model=self.model,
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                class_indices=class_indices,
                tokenizer=self.tokenizer,
                device=self.device,
            )
        finally:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.zero_grad()
        return sal
 
    # Predict 
 
    def predict(
        self,
        text:         str,
        window:       int = 1,
        top_k_tokens: int = 10,
    ) -> DocumentResult:
        sentences = split_sentences(text, tokenizer=self.tokenizer, max_tokens=self.max_length)
        if not sentences:
            return self._empty(text)
 
        sent_results: List[SentenceResult] = []
 
        for idx, sent in enumerate(sentences):
            # Sliding-window context
            if window == 0:
                ctx = sent
            else:
                left  = sentences[max(0, idx - window): idx]
                right = sentences[idx + 1: idx + 1 + window]
                ctx   = " ".join(left + [sent] + right)
 
            # Forward pass
            probs         = self._forward_probs(ctx)       # (7,)
            detected_mask = probs >= self.threshold         # (7,) bool
 
            # no class exceeds threshold → fallback neutral
            if not detected_mask.any():
                detected_mask[NEUTRAL_IDX] = True
 
            emotion_scores = [
                EmotionScore(
                    emotion=CLASS_NAMES[i],
                    probability=float(probs[i]),
                    detected=bool(detected_mask[i]),
                )
                for i in range(len(CLASS_NAMES))
            ]
 
            sent_is_neutral = (
                bool(detected_mask[NEUTRAL_IDX]) and
                not detected_mask[:NEUTRAL_IDX].any()
            )
 
            # Saliency only for detected emotion classes (excluding neutral)
            saliency_map: Dict[str, List[TokenSaliency]] = {}
            if self.compute_saliency and not sent_is_neutral:
                emo_indices = [i for i in range(NEUTRAL_IDX) if detected_mask[i]]
                if emo_indices:
                    raw_sal = self._saliency(ctx, emo_indices)
                    saliency_map = {
                        CLASS_NAMES[i]: raw_sal[i] for i in emo_indices
                    }
 
            sent_results.append(SentenceResult(
                sentence_idx=idx,
                sentence_text=sent,
                context_text=ctx,
                emotions=emotion_scores,
                token_saliency=saliency_map,
                is_neutral=sent_is_neutral,
            ))
 
        return self._aggregate(text, sent_results, top_k_tokens)
 
    # Aggregate 
 
    def _aggregate(
        self,
        full_text:    str,
        sent_results: List[SentenceResult],
        top_k:        int,
    ) -> DocumentResult:
        max_prob = {c: 0.0 for c in CLASS_NAMES}
        for sr in sent_results:
            for es in sr.emotions:
                max_prob[es.emotion] = max(max_prob[es.emotion], es.probability)
 
        detected = [
            emo for emo in EMOTION_NAMES
            if any(
                es.detected and es.emotion == emo
                for sr in sent_results
                for es in sr.emotions
            )
        ]
 
        agg: Dict[str, Dict[str, float]] = {emo: {} for emo in detected}
        for sr in sent_results:
            for emo, sal_list in sr.token_saliency.items():
                if emo not in agg:
                    continue
                for ts in sal_list:
                    key = ts.token.lower()
                    agg[emo][key] = max(agg[emo].get(key, 0.0), ts.score)
 
        top_tokens = {
            emo: sorted(agg[emo].items(), key=lambda x: x[1], reverse=True)[:top_k]
            for emo in detected
        }
 
        return DocumentResult(
            full_text=full_text,
            sentences=sent_results,
            detected_emotions=detected,
            emotion_confidence={c: round(max_prob[c], 4) for c in CLASS_NAMES},
            is_neutral=len(detected) == 0,
            top_tokens=top_tokens,
        )
 
    @staticmethod
    def _empty(text: str) -> DocumentResult:
        return DocumentResult(
            full_text=text,
            sentences=[],
            detected_emotions=[],
            emotion_confidence={c: 0.0 for c in CLASS_NAMES},
            is_neutral=True,
            top_tokens={},
        )
 

#  Entry point
 
if __name__ == "__main__":
    import json
 
    engine = EmotionInference(
        checkpoint_path=CHECKPOINT,
        config_path=CONFIG,
        compute_saliency=SALIENCY,
    )
 
    for text in TEXTS:
        result = engine.predict(text, window=WINDOW, top_k_tokens=TOP_K)
        if SHOW_JSON:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(result)
        print()
 