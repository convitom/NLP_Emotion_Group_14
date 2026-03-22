"""
grad_cam.py
===========
Text-domain analogue of Grad-CAM for transformer models
(ELECTRA, BERT, RoBERTa and any HuggingFace model with an embeddings block).

How it works vs image Grad-CAM
-------------------------------
Image Grad-CAM:
  1. Pick a convolutional feature map  A  (shape H×W×C).
  2. Compute gradient of class score w.r.t. every activation in A.
  3. Global-average-pool the gradients over the spatial H×W dims → weight α_c per channel.
  4. Weighted sum over channels + ReLU → heatmap of shape H×W.

Text Grad-CAM (this file):
  1. Pick the token embedding matrix  E  (shape L×D), one row per token.
     → analogue of the CNN feature map.
  2. Compute gradient of class score w.r.t. every element of E.
  3. Element-wise product  grad ⊙ E  then L2-norm over the hidden dim D → scalar per token.
     → analogue of "gradient × activation" weighting.
  4. Normalise to [0, 1] over the sequence → importance score per token.

Why gradient × embedding instead of gradient alone?
  Pure gradient shows sensitivity (what would change the score most if perturbed).
  Multiplying by the embedding itself keeps only the directions the model actually
  uses, filtering out spurious high-gradient dimensions where the embedding is near
  zero.  This mirrors the reasoning in Grad-CAM where α_c × A_c picks channels that
  are both strongly activated and strongly gradient-weighted.

Model-specific embedding layer names
--------------------------------------
  ELECTRA  → electra.embeddings        (word_embeddings inside)
  BERT     → bert.embeddings           (word_embeddings inside)
  RoBERTa  → roberta.embeddings        (word_embeddings inside)

The auto-detection below scans named_modules() for the first module whose name
ends with "embeddings" and has a child called "word_embeddings", so it works for
all three without hardcoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
#  Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenSaliency:
    """Importance score for a single (merged) token."""
    token:     str    # human-readable string (subwords already merged)
    score:     float  # normalised importance ∈ [0, 1]
    raw_score: float  # ‖grad ⊙ emb‖₂ before normalisation


# ─────────────────────────────────────────────────────────────────────────────
#  Embedding layer locator
# ─────────────────────────────────────────────────────────────────────────────

def _find_embedding_module(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """
    Return the top-level embeddings module for any BERT-family model.

    Search order:
      1. Module whose name ends with "embeddings" and contains word_embeddings
         (covers ELECTRA, BERT, RoBERTa, DeBERTa, DistilBERT …).
      2. Fallback: first nn.Embedding with vocab size > 1 000 (very safe net).
    """
    for name, mod in model.named_modules():
        if name.endswith("embeddings") and hasattr(mod, "word_embeddings"):
            return mod

    # Fallback: bare Embedding layer (less informative but won't crash)
    for mod in model.modules():
        if isinstance(mod, torch.nn.Embedding) and mod.weight.shape[0] > 1_000:
            return mod

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Subword token merger
# ─────────────────────────────────────────────────────────────────────────────

def _merge_subwords(
    tokens: List[str],
    scores: np.ndarray,
) -> Tuple[List[str], np.ndarray]:
    """
    Merge subword pieces into whole words, taking the max score per word.

    Handles three common tokenisation schemes:
      WordPiece  (BERT, ELECTRA):   continuation tokens start with "##"
      SentencePiece (RoBERTa, …):   word-start tokens start with "Ġ" or "▁"
      BPE without marker:           no reliable prefix — treated as WordPiece
    """
    merged_tokens: List[str]  = []
    merged_scores: List[float] = []

    for tok, sc in zip(tokens, scores.tolist()):
        if tok.startswith("##"):
            # WordPiece continuation — append to previous token
            if merged_tokens:
                merged_tokens[-1] += tok[2:]
                merged_scores[-1]  = max(merged_scores[-1], sc)
            else:
                merged_tokens.append(tok[2:])
                merged_scores.append(sc)
        elif tok.startswith("Ġ") or tok.startswith("▁"):
            # SentencePiece / RoBERTa word-start marker
            merged_tokens.append(tok[1:] or tok)
            merged_scores.append(sc)
        else:
            merged_tokens.append(tok)
            merged_scores.append(sc)

    return merged_tokens, np.array(merged_scores, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Core saliency computation
# ─────────────────────────────────────────────────────────────────────────────

# Special tokens to exclude from output (all lowercased for comparison)
_SPECIAL_TOKENS = {
    "[cls]", "[sep]", "[pad]", "[unk]", "[mask]",
    "<s>", "</s>", "<pad>", "<unk>", "<mask>",
}


def compute_token_saliency(
    model:          torch.nn.Module,
    input_ids:      torch.Tensor,       # (1, L)
    attention_mask: torch.Tensor,       # (1, L)
    class_idx:      int,
    tokenizer:      AutoTokenizer,
    device:         torch.device,
) -> List[TokenSaliency]:
    """
    Compute gradient × embedding saliency for one class.

    Parameters
    ----------
    model        : the stage-2 (or stage-1) classifier, already on `device`.
    input_ids    : tokenised input, shape (1, L).
    attention_mask: shape (1, L).
    class_idx    : index of the emotion class to explain.
    tokenizer    : matching tokenizer (for decoding token ids to strings).
    device       : torch device.

    Returns
    -------
    List[TokenSaliency] — one entry per non-special token in the input,
    sorted by position (not by score).
    """
    model.eval()

    # ── Locate embedding layer ────────────────────────────────────────────────
    emb_module = _find_embedding_module(model)

    # ── Hook: capture the embedding output tensor so we can backprop through it
    captured: List[torch.Tensor] = []

    def _fwd_hook(_module, _inp, out: torch.Tensor) -> torch.Tensor:
        # Detach from graph, re-attach with requires_grad so we can backprop
        out_with_grad = out.detach().requires_grad_(True)
        captured.clear()
        captured.append(out_with_grad)
        return out_with_grad   # replace the module's output

    handle = emb_module.register_forward_hook(_fwd_hook) if emb_module else None

    try:
        # ── Forward pass ─────────────────────────────────────────────────────
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        model.zero_grad()
        logits = model(input_ids, attention_mask)           # (1, num_classes)
        score  = torch.sigmoid(logits[0, class_idx])        # scalar ∈ (0,1)

        # ── Backward ─────────────────────────────────────────────────────────
        score.backward()

        # ── Extract saliency ─────────────────────────────────────────────────
        if captured and captured[0].grad is not None:
            emb  = captured[0]                              # (1, L, D)
            grad = captured[0].grad                         # (1, L, D)
            # Grad-CAM analogue: element-wise product then L2 norm over D
            raw_scores = (grad * emb).norm(dim=-1).squeeze(0)  # (L,)
        else:
            # Fallback if hook didn't fire (should not happen in practice)
            raw_scores = torch.zeros(input_ids.shape[1], device=device)

    finally:
        if handle:
            handle.remove()
        model.zero_grad()

    raw_scores = raw_scores.detach().cpu().float().numpy()

    # ── Decode tokens ─────────────────────────────────────────────────────────
    seq_len    = int(attention_mask[0].sum().item())
    token_ids  = input_ids[0, :seq_len].cpu().tolist()
    tokens     = tokenizer.convert_ids_to_tokens(token_ids)
    raw_scores = raw_scores[:seq_len]

    # ── Merge subword pieces ──────────────────────────────────────────────────
    merged_tokens, merged_scores = _merge_subwords(tokens, raw_scores)

    # ── Normalise to [0, 1] ───────────────────────────────────────────────────
    max_val = merged_scores.max()
    norm_scores = merged_scores / max_val if max_val > 1e-9 else merged_scores.copy()

    # ── Build output, skip special tokens ────────────────────────────────────
    result: List[TokenSaliency] = []
    for tok, ns, rs in zip(merged_tokens, norm_scores.tolist(), merged_scores.tolist()):
        if not tok or tok.strip() == "" or tok.lower() in _SPECIAL_TOKENS:
            continue
        result.append(TokenSaliency(token=tok, score=ns, raw_score=rs))

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Batch helper — compute saliency for multiple classes in one call
# ─────────────────────────────────────────────────────────────────────────────

def compute_saliency_for_classes(
    model:          torch.nn.Module,
    input_ids:      torch.Tensor,
    attention_mask: torch.Tensor,
    class_indices:  List[int],
    tokenizer:      AutoTokenizer,
    device:         torch.device,
) -> Dict[int, List[TokenSaliency]]:
    """
    Convenience wrapper: compute saliency for several classes at once.

    Returns dict mapping class_idx → List[TokenSaliency].
    Each class requires a separate backward pass (independent computation graphs).
    """
    results: Dict[int, List[TokenSaliency]] = {}
    for idx in class_indices:
        results[idx] = compute_token_saliency(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            class_idx=idx,
            tokenizer=tokenizer,
            device=device,
        )
    return results
