"""
Masked marginal scoring — Track A (refs/fair-esm @ 2b369911, fp32).

Score = Σ_i∈mut [ log p(x*_i | x_{-i}) − log p(x_i | x_{-i}) ]

Uses the fair-esm alphabet API directly:
  model(tokens)["logits"]  shape: (1, L+2, vocab)
  alphabet.mask_idx        integer mask token id
  alphabet.get_idx(aa)     single amino-acid vocab lookup

Cost: |unique positions| forward passes per assay.
Long sequences (L > 1022): optimal window centred on mutation position.

Source: Meier et al. NeurIPS 2021; ProteinGym compute_fitness.py strategy="masked-marginals"
"""

import torch


@torch.no_grad()
def score_variants(
    model,
    alphabet,
    sequence: str,
    variants: list[str],
    device: str = "cuda",
) -> list[float]:
    model.eval()
    L = len(sequence)
    mask_id = alphabet.mask_idx
    batch_converter = alphabet.get_batch_converter()

    # Collect unique positions needed
    positions_needed: set[int] = set()
    parsed: list[list[tuple[str, int, str]]] = []
    for variant_str in variants:
        muts = []
        for mut in variant_str.split(":"):
            wt_aa, pos_1, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
            muts.append((wt_aa, pos_1, mut_aa))
            positions_needed.add(pos_1)
        parsed.append(muts)

    # One forward pass per unique masked position
    log_probs: dict[int, torch.Tensor] = {}
    for pos_1 in sorted(positions_needed):
        if L <= 1022:
            lp = _masked_pass(model, batch_converter, sequence, pos_1, mask_id, device)
        else:
            lp = _windowed_masked_pass(model, batch_converter, sequence, pos_1, mask_id, device)
        log_probs[pos_1] = lp

    scores = []
    for muts in parsed:
        score = 0.0
        for wt_aa, pos_1, mut_aa in muts:
            wt_id  = alphabet.get_idx(wt_aa)
            mut_id = alphabet.get_idx(mut_aa)
            lp = log_probs[pos_1]
            score += (lp[mut_id] - lp[wt_id]).item()
        scores.append(score)
    return scores


def _masked_pass(model, batch_converter, sequence, pos_1, mask_id, device):
    _, _, tokens = batch_converter([("seq", sequence)])
    tokens = tokens.to(device)
    tokens[0, pos_1] = mask_id          # pos_1: BOS at 0, AA_1 at 1 .. AA_L at L
    logits = model(tokens)["logits"][0, pos_1]   # (vocab,)
    return torch.log_softmax(logits.float(), dim=-1)


def _windowed_masked_pass(model, batch_converter, sequence, pos_1, mask_id, device, window=1022):
    """Optimal window centred on mutation. Matches ProteinGym 'optimal' windowing strategy."""
    L = len(sequence)
    half  = window // 2
    pos_0 = pos_1 - 1  # 0-indexed in sequence string

    if pos_0 < half:
        start, end = 0, window
    elif pos_0 >= L - half:
        start, end = L - window, L
    else:
        start, end = pos_0 - half, pos_0 + half

    start, end = max(0, start), min(L, end)
    chunk      = sequence[start:end]
    local_tok  = (pos_0 - start) + 1  # +1 for BOS

    _, _, tokens = batch_converter([("seq", chunk)])
    tokens = tokens.to(device)
    tokens[0, local_tok] = mask_id
    logits = model(tokens)["logits"][0, local_tok]
    return torch.log_softmax(logits.float(), dim=-1)
