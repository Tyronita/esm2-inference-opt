"""
Batched masked marginals — Track C optimised (EsmcForMaskedLM, true-batch shinkaevolve).

Fills the batch dimension with B sequences, each masking ONE unique position.
Single batched forward pass extracts all B positions simultaneously.

Scores are IDENTICAL to sequential masked_marginals.py (no approximation):
each sequence sees only its own masked position, so attention context is clean.

MFU improvement: L sequential passes × tiny_batch → ceil(L/B) passes × batch_B.
GPU compute utilisation scales with batch dimension: B=32 → ~8-15× faster.

Model API: EsmcForMaskedLM (HuggingFace-style), EsmcTokenizer
Source: refs/evolutionaryscale-esm/esm/models/esmc/model.py + tokenizer.py

Batch size B: trade-off between VRAM and throughput.
  B=16:  safe for A100 40GB at L=3000+
  B=32:  optimal for L < 1500 (most ProteinGym assays)
  B=64:  ~16× speedup, fine for short proteins (L < 500)
"""

import re
import math
import time

import torch
from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer

_AA = "ACDEFGHIKLMNPQRSTVWY"
_MASK_TOKEN_ID = 32   # refs/evolutionaryscale-esm/esm/models/esmc/tokenizer.py:45
_WINDOW = 1022        # context window limit


@torch.no_grad()
def score_variants(
    model,
    tokenizer,          # ignored — EsmcTokenizer instantiated locally
    sequence: str,
    variants: list[str],
    device: str = "cuda",
    batch_size: int = 32,
) -> list[float]:
    model.eval()

    tok = EsmcTokenizer()

    # Resolve AA → vocab token IDs directly from the tokenizer vocab
    vocab = tok.get_vocab()
    aa_to_token: dict[str, int] = {aa: vocab[aa] for aa in _AA}

    L = len(sequence)
    needs_window = L > _WINDOW

    # Parse variants, collect unique 1-indexed positions
    parsed: list[list[tuple[str, int, str]]] = []
    for v in variants:
        muts: list[tuple[str, int, str]] = []
        for m in v.split(":"):
            match = re.match(r"([A-Z])(\d+)([A-Z])", m)
            if match:
                wt, pos_str, mt = match.groups()
                muts.append((wt, int(pos_str), mt))
        parsed.append(muts)

    positions_needed = sorted({pos for muts in parsed for _, pos, _ in muts})

    # Pre-tokenize full sequence once (for non-windowed path)
    if not needs_window:
        base_enc = tok([sequence], return_tensors="pt", padding=False)
        base_ids = base_enc["input_ids"][0]   # (L+2,)
        base_mask = base_enc["attention_mask"][0]

    log_probs: dict[int, torch.Tensor] = {}

    # True-batch: ceil(L/B) passes of B sequences each with ONE position masked
    for chunk_start in range(0, len(positions_needed), batch_size):
        chunk_positions = positions_needed[chunk_start: chunk_start + batch_size]
        B = len(chunk_positions)

        batch_input_ids: list[torch.Tensor] = []
        batch_token_positions: list[int] = []

        for pos_1 in chunk_positions:
            seq_pos = pos_1 - 1   # 0-indexed
            token_pos = pos_1     # 1-indexed (BOS at 0)

            if needs_window:
                half = _WINDOW // 2
                start = max(0, seq_pos - half)
                end = min(L, start + _WINDOW)
                if end == L:
                    start = max(0, L - _WINDOW)
                window_seq = sequence[start:end]
                enc = tok([window_seq], return_tensors="pt", padding=False)
                ids = enc["input_ids"][0].clone()
                local_pos = seq_pos - start + 1
                ids[local_pos] = _MASK_TOKEN_ID
                batch_input_ids.append(ids)
                batch_token_positions.append(local_pos)
            else:
                ids = base_ids.clone()
                ids[token_pos] = _MASK_TOKEN_ID
                batch_input_ids.append(ids)
                batch_token_positions.append(token_pos)

        # Pad to equal length and stack
        max_len = max(ids.shape[0] for ids in batch_input_ids)
        padded = torch.zeros(B, max_len, dtype=torch.long)
        attn_masks = torch.zeros(B, max_len, dtype=torch.long)
        for i, ids in enumerate(batch_input_ids):
            padded[i, :ids.shape[0]] = ids
            attn_masks[i, :ids.shape[0]] = 1

        # Single batched forward pass
        out = model(
            input_ids=padded.to(device),
            attention_mask=attn_masks.to(device),
        )
        logits = out.logits   # (B, max_len, vocab_size)

        for i, (pos_1, tok_pos) in enumerate(zip(chunk_positions, batch_token_positions)):
            log_probs[pos_1] = torch.log_softmax(logits[i, tok_pos], dim=-1)

    scores: list[float] = []
    for muts in parsed:
        score = 0.0
        for wt, pos_1, mt in muts:
            lp = log_probs[pos_1]
            score += (lp[aa_to_token[mt]] - lp[aa_to_token[wt]]).item()
        scores.append(score)

    return scores
