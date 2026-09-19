"""
Pseudo-perplexity scoring — Track A (refs/fair-esm @ 2b369911, fp32).

For each variant: build mutant sequence, then score masked marginals on the mutant.
  score = Σ_i∈mut [ log p(x*_i | mutant_{-i}) − log p(x_i | mutant_{-i}) ]

For single-site variants this is numerically identical to masked_marginals
(only one position mutated, so context is the same either way).

Cost: |unique positions| forward passes per variant (L passes per assay for single-site).

Source: Meier et al. NeurIPS 2021 / ProteinGym compute_fitness.py strategy="pseudo-ppl"
"""

import torch

from scoring.ref.masked_marginals import _masked_pass, _windowed_masked_pass


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

    scores = []
    for variant_str in variants:
        muts = []
        for mut in variant_str.split(":"):
            muts.append((mut[0], int(mut[1:-1]), mut[-1]))

        # Build mutant sequence
        seq_list = list(sequence)
        for _, pos_1, mut_aa in muts:
            seq_list[pos_1 - 1] = mut_aa
        mutant_seq = "".join(seq_list)

        score = 0.0
        for wt_aa, pos_1, mut_aa in muts:
            if L <= 1022:
                lp = _masked_pass(model, batch_converter, mutant_seq, pos_1, mask_id, device)
            else:
                lp = _windowed_masked_pass(model, batch_converter, mutant_seq, pos_1, mask_id, device)

            wt_id  = alphabet.get_idx(wt_aa)
            mut_id = alphabet.get_idx(mut_aa)
            score += (lp[mut_id] - lp[wt_id]).item()
        scores.append(score)

    return scores
