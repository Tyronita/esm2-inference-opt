# =============================================================================
# TRACK B — transformers==4.44.0, fp16, HuggingFace API
# STATUS: commented out for benchmark runs. Kept for:
#   (a) check_consistency.py — the consistency gate (rho=+0.999973 vs Track A)
#   (b) future SOTA optimisation work (Flash Attention 2, torch.compile, etc.)
# DO NOT use for ProteinGym benchmark scoring — use scoring/ref/ (Track A) instead.
# TODO: when upgrading to SOTA settings, reactivate and update API calls here.
# =============================================================================
"""
Masked marginal log-likelihood ratio scoring.

TRACK B: transformers==4.44.0 HuggingFace API (fp16).
For the reference fp32 fair-esm implementation used in benchmarks, see scoring/ref/masked_marginals.py.

Source: Meier et al. NeurIPS 2021 (ESM-1v paper), §Scoring Methods
ProteinGym: compute_fitness.py strategy="masked-marginals"

Score = Σ_mutated_i [ log p(mut_i | wt_{-i}) - log p(wt_i | wt_{-i}) ]

where wt_{-i} is the wildtype sequence with position i replaced by <mask>.

Cost: L forward passes per assay (one mask per position).

Limitations:
  - O(L) forward passes — LRRK2 (L=2527) requires 2527 passes
  - For L > 1022 AA: optimal windowing used (1024-token window centred on mutation)
  - Cannot batch variants easily without masking multiple positions simultaneously
"""

import torch


@torch.no_grad()
def score_variants(
    model,
    tokenizer,
    sequence: str,
    variants: list[str],
    device: str = "cuda",
) -> list[float]:
    model.eval()
    L = len(sequence)
    mask_id = tokenizer.mask_token_id

    # Collect all unique positions that need to be masked
    positions_needed: set[int] = set()
    parsed: list[list[tuple[str, int, str]]] = []
    for variant_str in variants:
        muts = []
        for mut in variant_str.split(":"):
            wt_aa, pos_1, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
            muts.append((wt_aa, pos_1, mut_aa))
            positions_needed.add(pos_1)
        parsed.append(muts)

    # One forward pass per unique position
    log_probs: dict[int, torch.Tensor] = {}
    for pos_1 in sorted(positions_needed):
        if L <= 1022:
            lp = _masked_pass(model, tokenizer, sequence, pos_1, mask_id, device)
        else:
            lp = _windowed_masked_pass(model, tokenizer, sequence, pos_1, mask_id, device)
        log_probs[pos_1] = lp

    scores = []
    for muts in parsed:
        score = 0.0
        for wt_aa, pos_1, mut_aa in muts:
            wt_id  = tokenizer.convert_tokens_to_ids(wt_aa)
            mut_id = tokenizer.convert_tokens_to_ids(mut_aa)
            lp = log_probs[pos_1]
            score += (lp[mut_id] - lp[wt_id]).item()
        scores.append(score)
    return scores


def _masked_pass(model, tokenizer, sequence, pos_1, mask_id, device):
    enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    ids = enc["input_ids"].clone()
    ids[0, pos_1] = mask_id          # pos_1 maps directly: BOS at 0, AA at 1..L
    logits = model(input_ids=ids).logits[0, pos_1]   # (vocab,)
    return torch.log_softmax(logits.float(), dim=-1)


def _windowed_masked_pass(model, tokenizer, sequence, pos_1, mask_id, device, window=1022):
    """Optimal window centred on mutation position. Matches ProteinGym 'optimal' strategy."""
    L = len(sequence)
    half = window // 2
    pos_0 = pos_1 - 1   # 0-indexed in sequence string

    if pos_0 < half:
        start, end = 0, window
    elif pos_0 >= L - half:
        start, end = L - window, L
    else:
        start, end = pos_0 - half, pos_0 + half

    start, end = max(0, start), min(L, end)
    chunk = sequence[start:end]
    local_pos = pos_0 - start  # 0-indexed within chunk
    local_tok  = local_pos + 1  # +1 for BOS

    enc = tokenizer(chunk, return_tensors="pt", add_special_tokens=True).to(device)
    ids = enc["input_ids"].clone()
    ids[0, local_tok] = mask_id
    logits = model(input_ids=ids).logits[0, local_tok]
    return torch.log_softmax(logits.float(), dim=-1)
