"""
Simple OFS — naive single-pass approximation.

WARNING: This is NOT the Kantroo et al. 2024 OFS method.

True OFS (arXiv:2407.07265) uses an ensemble of 8 trained MLPs that map
unmasked residue embeddings → predicted masked probability distributions.
Those MLP heads are trained on ESM-2 650M and are not reproduced here.

What this implements: use the unmasked forward pass logits directly at each
position. This is functionally equivalent to wt_marginals — the model has seen
position i when computing its logit there. It is included only to make the
equivalence explicit and document the limitation.

Expected result: Spearman ρ ≈ wt_marginals (both ~0.40, below masked_marginals ~0.44).

To use true OFS: see https://github.com/prkaavya/ofsprotein (if released)
or Kantroo, Wagner & Machta (2024), PRX Life, arXiv:2407.07265.
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

    enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    logits = model(input_ids=enc["input_ids"]).logits[0]  # (L+2, vocab)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    scores = []
    for variant_str in variants:
        score = 0.0
        for mut in variant_str.split(":"):
            wt_aa, pos_1, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
            wt_id  = tokenizer.convert_tokens_to_ids(wt_aa)
            mut_id = tokenizer.convert_tokens_to_ids(mut_aa)
            score += (log_probs[pos_1, mut_id] - log_probs[pos_1, wt_id]).item()
        scores.append(score)
    return scores
