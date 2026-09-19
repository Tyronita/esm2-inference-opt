"""
Wildtype marginal scoring — single forward pass, lower bound on accuracy.

Score = log p(mut | wt_context) - log p(wt | wt_context)
where context is the FULL UNMASKED wildtype sequence.

Cheaper than masked marginal (1 pass vs L passes) but less accurate
because it doesn't isolate position-specific uncertainty.
Included as a fast sanity-check baseline.
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
    model = model.to(device).eval()

    enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    logits = model(input_ids=enc["input_ids"]).logits[0]  # (L+2, vocab)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    scores = []
    for variant_str in variants:
        score = 0.0
        for mut in variant_str.split(":"):
            wt_aa  = mut[0]
            mut_aa = mut[-1]
            pos    = int(mut[1:-1])
            wt_id  = tokenizer.convert_tokens_to_ids(wt_aa)
            mut_id = tokenizer.convert_tokens_to_ids(mut_aa)
            score += (log_probs[pos, mut_id] - log_probs[pos, wt_id]).item()
        scores.append(score)

    return scores
