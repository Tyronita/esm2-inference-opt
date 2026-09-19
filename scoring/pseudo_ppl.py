# =============================================================================
# TRACK B — transformers==4.44.0, fp16, HuggingFace API
# STATUS: commented out for benchmark runs. See scoring/ref/pseudo_ppl.py for Track A.
# =============================================================================
"""
Pseudo-perplexity scoring on the mutant sequence.

TRACK B: transformers==4.44.0 HuggingFace API (fp16).
For the reference implementation used in benchmarks, see scoring/ref/pseudo_ppl.py.

Source: Meier et al. NeurIPS 2021 / ProteinGym compute_fitness.py strategy="pseudo-ppl"

For each variant, create the mutant sequence, then compute masked marginals on it:
  score = Σ_mutated_i [ log p(mut_i | mutant_{-i}) - log p(wt_i | mutant_{-i}) ]

The key difference from masked_marginals: the *context* used is the mutant sequence,
not the wildtype. For single-site variants this is identical to masked_marginals
(masking the only mutated position means the context IS the same either way).
For multi-site variants the contexts differ.

Cost: L forward passes per assay (same as masked_marginals).
Note: ProteinGym benchmarks are dominated by single-site variants, so in practice
pseudo_ppl ≈ masked_marginals on most assays.
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

    scores = []
    for variant_str in variants:
        muts = []
        for mut in variant_str.split(":"):
            muts.append((mut[0], int(mut[1:-1]), mut[-1]))

        # Build mutant sequence
        seq_list = list(sequence)
        for wt_aa, pos_1, mut_aa in muts:
            seq_list[pos_1 - 1] = mut_aa
        mutant_seq = "".join(seq_list)

        score = 0.0
        for wt_aa, pos_1, mut_aa in muts:
            if L <= 1022:
                lp = _masked_pass(model, tokenizer, mutant_seq, pos_1, mask_id, device)
            else:
                from scoring.masked_marginals import _windowed_masked_pass
                lp = _windowed_masked_pass(model, tokenizer, mutant_seq, pos_1, mask_id, device)

            wt_id  = tokenizer.convert_tokens_to_ids(wt_aa)
            mut_id = tokenizer.convert_tokens_to_ids(mut_aa)
            score += (lp[mut_id] - lp[wt_id]).item()
        scores.append(score)

    return scores


def _masked_pass(model, tokenizer, sequence, pos_1, mask_id, device):
    enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    ids = enc["input_ids"].clone()
    ids[0, pos_1] = mask_id
    logits = model(input_ids=ids).logits[0, pos_1]
    return torch.log_softmax(logits.float(), dim=-1)
