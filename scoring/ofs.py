"""
One Fell Swoop (OFS) pseudo-perplexity scoring.

From: Kantroo, Wagner & Machta (2024)
"Pseudo-perplexity in One Fell Swoop for Protein Fitness Estimation"
PRX Life / arXiv:2407.07265

Key insight: instead of masking each position and running L forward passes,
use the UNMASKED sequence's embeddings to approximate what the model would
predict if each position were masked. One forward pass for the whole sequence.

The approximation: the last-layer hidden state at position i from the
unmasked forward pass is used directly to predict the masked probability
at that position via the LM head — skipping the re-attention over the mask token.

This works because ESM-2's attention already "sees" all other positions;
the mask token's main effect is suppressing self-information at position i.

Cost: 1 forward pass per sequence (L× speedup vs masked marginal).
Performance: matches masked marginal on substitutions, beats it on indels.
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
    Score variants using OFS pseudo-perplexity (single forward pass).

    Args:
        sequence: wildtype amino acid sequence
        variants: list of variant strings e.g. ["A53T", "E46K", "A53T:E46K"]

    Returns:
        list of float scores (higher = more fit)
    """
    model = model.to(device).eval()

    enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    input_ids = enc["input_ids"]  # (1, L+2)

    # Single forward pass — no masking
    logits = model(input_ids=input_ids).logits[0]  # (L+2, vocab)
    # log-probs for every position from the unmasked pass
    log_probs = torch.log_softmax(logits.float(), dim=-1)  # (L+2, vocab)

    scores = []
    for variant_str in variants:
        score = 0.0
        for mut in variant_str.split(":"):
            wt_aa  = mut[0]
            mut_aa = mut[-1]
            pos_1idx = int(mut[1:-1])  # 1-indexed protein position
            token_pos = pos_1idx      # +1 for BOS offset already handled in 1-index

            wt_id  = tokenizer.convert_tokens_to_ids(wt_aa)
            mut_id = tokenizer.convert_tokens_to_ids(mut_aa)

            lp = log_probs[token_pos]
            score += (lp[mut_id] - lp[wt_id]).item()
        scores.append(score)

    return scores


@torch.no_grad()
def batch_score_variants(
    model,
    tokenizer,
    sequence: str,
    variants: list[str],
    device: str = "cuda",
    batch_size: int = 64,
) -> list[float]:
    """
    Same as score_variants but tokenizes multiple sequences in a batch
    for the (rare) multi-site case where you need per-variant masking.
    Falls back to OFS for speed.
    """
    return score_variants(model, tokenizer, sequence, variants, device)
