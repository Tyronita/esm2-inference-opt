"""
Masked marginal log-likelihood ratio scoring.

Canonical method from Meier et al. NeurIPS 2021 (ESM-1v paper).
Cost: L forward passes per sequence (one mask per position).

Score for variant X_i -> Y_i at position i:
    score = log p(Y_i | x_{\\i}) - log p(X_i | x_{\\i})

For multi-site variants: sum over all mutated positions.
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
    """
    Score a list of variant strings against a wildtype sequence.

    Args:
        sequence: wildtype amino acid sequence (no special tokens)
        variants: list of strings like "A53T" or "A53T:E46K" (multi-site colon-separated)
        device: "cuda" | "cpu"

    Returns:
        list of float scores (higher = more fit relative to wildtype)
    """
    model = model.to(device).eval()

    # Tokenize wildtype once to get token IDs
    enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    input_ids = enc["input_ids"]  # (1, L+2)  BOS + seq + EOS

    L = len(sequence)
    mask_id = tokenizer.mask_token_id

    # One forward pass per position — cache log-probs for all positions
    # Shape: log_probs[i] = log_softmax over vocab at position i (1-indexed, skipping BOS)
    log_probs = {}
    for pos in range(1, L + 1):  # 1-indexed in token space (BOS is 0)
        masked = input_ids.clone()
        masked[0, pos] = mask_id
        logits = model(input_ids=masked).logits[0, pos]  # (vocab,)
        log_probs[pos] = torch.log_softmax(logits.float(), dim=-1)

    # Score each variant
    scores = []
    for variant_str in variants:
        score = 0.0
        for mut in variant_str.split(":"):
            wt_aa = mut[0]
            mut_aa = mut[-1]
            pos_1idx = int(mut[1:-1])          # 1-indexed protein position
            token_pos = pos_1idx               # token pos = protein pos (BOS shifts by +1 but we already +1'd above)

            wt_id  = tokenizer.convert_tokens_to_ids(wt_aa)
            mut_id = tokenizer.convert_tokens_to_ids(mut_aa)

            lp = log_probs[token_pos]
            score += (lp[mut_id] - lp[wt_id]).item()
        scores.append(score)

    return scores
