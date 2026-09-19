"""
Batched masked marginals — mask B positions simultaneously per forward pass.

This work (optimization of masked_marginals).

Instead of one mask per pass (L passes total), mask B positions per pass,
reducing to ceil(L/B) passes. Trades speed for approximation accuracy:
masking multiple positions simultaneously changes the attention context
each masked position sees, introducing small errors vs single-position masking.

B=1  → identical to masked_marginals (exact, L passes)
B=L  → all positions masked simultaneously (equivalent to computing
        log p of the full sequence under fully-masked context — degenerate)
B=8  → good balance: ~8× speedup, small approximation error on typical proteins
B=32 → ~32× speedup, larger error

Limitation: the masked tokens see each other as [MASK] in attention.
With B large, positions near multiple masks get corrupted context.
"""

import math

import torch


@torch.no_grad()
def score_variants(
    model,
    tokenizer,
    sequence: str,
    variants: list[str],
    device: str = "cuda",
    batch_size: int = 8,
) -> list[float]:
    model.eval()
    L = len(sequence)
    mask_id = tokenizer.mask_token_id

    # Collect unique positions
    positions_needed: set[int] = set()
    parsed: list[list[tuple[str, int, str]]] = []
    for variant_str in variants:
        muts = []
        for mut in variant_str.split(":"):
            wt_aa, pos_1, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
            muts.append((wt_aa, pos_1, mut_aa))
            positions_needed.add(pos_1)
        parsed.append(muts)

    sorted_positions = sorted(positions_needed)
    log_probs: dict[int, torch.Tensor] = {}

    # Tokenize once
    enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    base_ids = enc["input_ids"]  # (1, L+2)

    # Process in batches of B positions
    for i in range(0, len(sorted_positions), batch_size):
        batch_pos = sorted_positions[i: i + batch_size]

        # Single forward pass with all B positions masked
        ids = base_ids.clone()
        for p in batch_pos:
            ids[0, p] = mask_id

        logits = model(input_ids=ids).logits[0]  # (L+2, vocab)

        for p in batch_pos:
            log_probs[p] = torch.log_softmax(logits[p].float(), dim=-1)

    scores = []
    for muts in parsed:
        score = 0.0
        for wt_aa, pos_1, mut_aa in muts:
            wt_id  = tokenizer.convert_tokens_to_ids(wt_aa)
            mut_id = tokenizer.convert_tokens_to_ids(mut_aa)
            score += (log_probs[pos_1][mut_id] - log_probs[pos_1][wt_id]).item()
        scores.append(score)
    return scores
