from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

@dataclass
class TokenSaliency:
    token:     str    
    score:     float  
    raw_score: float  

def _find_embedding_module(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    for name, mod in model.named_modules():
        if name.endswith("embeddings") and hasattr(mod, "word_embeddings"):
            return mod

    # Fallback: bare Embedding layer (less informative but won't crash)
    for mod in model.modules():
        if isinstance(mod, torch.nn.Embedding) and mod.weight.shape[0] > 1_000:
            return mod

    return None

def _merge_subwords(
    tokens: List[str],
    scores: np.ndarray,
) -> Tuple[List[str], np.ndarray]:
    merged_tokens: List[str]  = []
    merged_scores: List[float] = []

    for tok, sc in zip(tokens, scores.tolist()):
        if tok.startswith("##"):
            if merged_tokens:
                merged_tokens[-1] += tok[2:]
                merged_scores[-1]  = max(merged_scores[-1], sc)
            else:
                merged_tokens.append(tok[2:])
                merged_scores.append(sc)
        elif tok.startswith("Ġ") or tok.startswith("▁"):
            merged_tokens.append(tok[1:] or tok)
            merged_scores.append(sc)
        else:
            merged_tokens.append(tok)
            merged_scores.append(sc)

    return merged_tokens, np.array(merged_scores, dtype=np.float32)


_SPECIAL_TOKENS = {
    "[cls]", "[sep]", "[pad]", "[unk]", "[mask]",
    "<s>", "</s>", "<pad>", "<unk>", "<mask>",
}


def compute_token_saliency(
    model:          torch.nn.Module,
    input_ids:      torch.Tensor,       
    attention_mask: torch.Tensor,       
    class_idx:      int,
    tokenizer:      AutoTokenizer,
    device:         torch.device,
) -> List[TokenSaliency]:
    model.eval()

  
    emb_module = _find_embedding_module(model)

  
    captured: List[torch.Tensor] = []

    def _fwd_hook(_module, _inp, out: torch.Tensor) -> torch.Tensor:
        out_with_grad = out.detach().requires_grad_(True)
        captured.clear()
        captured.append(out_with_grad)
        return out_with_grad  

    handle = emb_module.register_forward_hook(_fwd_hook) if emb_module else None

    try:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        model.zero_grad()
        logits = model(input_ids, attention_mask)          
        score  = torch.sigmoid(logits[0, class_idx])        
        score.backward()
        if captured and captured[0].grad is not None:
            emb  = captured[0]                             
            grad = captured[0].grad                         
            raw_scores = (grad * emb).norm(dim=-1).squeeze(0)  
        else:
            raw_scores = torch.zeros(input_ids.shape[1], device=device)

    finally:
        if handle:
            handle.remove()
        model.zero_grad()

    raw_scores = raw_scores.detach().cpu().float().numpy()

    seq_len    = int(attention_mask[0].sum().item())
    token_ids  = input_ids[0, :seq_len].cpu().tolist()
    tokens     = tokenizer.convert_ids_to_tokens(token_ids)
    raw_scores = raw_scores[:seq_len]

    merged_tokens, merged_scores = _merge_subwords(tokens, raw_scores)
    max_val = merged_scores.max()
    norm_scores = merged_scores / max_val if max_val > 1e-9 else merged_scores.copy()

    result: List[TokenSaliency] = []
    for tok, ns, rs in zip(merged_tokens, norm_scores.tolist(), merged_scores.tolist()):
        if not tok or tok.strip() == "" or tok.lower() in _SPECIAL_TOKENS:
            continue
        result.append(TokenSaliency(token=tok, score=ns, raw_score=rs))

    return result


def compute_saliency_for_classes(
    model:          torch.nn.Module,
    input_ids:      torch.Tensor,
    attention_mask: torch.Tensor,
    class_indices:  List[int],
    tokenizer:      AutoTokenizer,
    device:         torch.device,
) -> Dict[int, List[TokenSaliency]]:
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
