"""
Wildtype marginal scoring — Track A (refs/fair-esm @ 2b369911, fp32).

Score = log p(mut_aa | full_wt_sequence) − log p(wt_aa | full_wt_sequence)

Cost: 1 forward pass per assay.
Long sequences (L > 1022): overlapping sigmoid-weighted windows.

Source: Meier et al. NeurIPS 2021; ProteinGym compute_fitness.py strategy="wt-marginals"
"""

import math

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
    batch_converter = alphabet.get_batch_converter()

    if L <= 1022:
        log_probs = _single_pass(model, batch_converter, sequence, device)
    else:
        log_probs = _windowed_pass(model, batch_converter, sequence, device, alphabet)

    scores = []
    for variant_str in variants:
        score = 0.0
        for mut in variant_str.split(":"):
            wt_aa, pos_1, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
            wt_id  = alphabet.get_idx(wt_aa)
            mut_id = alphabet.get_idx(mut_aa)
            score += (log_probs[pos_1, mut_id] - log_probs[pos_1, wt_id]).item()
        scores.append(score)
    return scores


def _single_pass(model, batch_converter, sequence, device):
    _, _, tokens = batch_converter([("seq", sequence)])
    logits = model(tokens.to(device))["logits"][0]       # (L+2, vocab)
    return torch.log_softmax(logits.float(), dim=-1)


def _windowed_pass(model, batch_converter, sequence, device, alphabet, window=1022, overlap=256):
    """Overlapping sigmoid-weighted windows. Matches ProteinGym 'overlapping' strategy."""
    L = len(sequence)
    vocab = len(alphabet)
    log_probs  = torch.full((L + 2, vocab), float("-inf"), device=device)
    weight_sum = torch.zeros(L + 2, device=device)

    start = 0
    while start < L:
        end   = min(start + window, L)
        chunk = sequence[start:end]
        _, _, tokens = batch_converter([("seq", chunk)])
        lp = torch.log_softmax(
            model(tokens.to(device))["logits"][0].float(), dim=-1
        )  # (chunk_len+2, vocab)

        chunk_L   = end - start
        positions = torch.arange(chunk_L, dtype=torch.float32, device=device)
        weights   = torch.sigmoid((positions - overlap) / 10) * torch.sigmoid(
            (chunk_L - 1 - positions - overlap) / 10
        )
        weights = weights.clamp(min=1e-6)

        for local_i, global_i in enumerate(range(start + 1, end + 1)):
            w    = weights[local_i]
            prev = log_probs[global_i]
            if prev.isinf().all():
                log_probs[global_i]  = lp[local_i + 1]
                weight_sum[global_i] = w
            else:
                log_probs[global_i] = torch.logaddexp(
                    prev + math.log(weight_sum[global_i].item()),
                    lp[local_i + 1] + math.log(w.item()),
                ) - math.log(weight_sum[global_i].item() + w.item())
                weight_sum[global_i] += w

        start += window - overlap

    return log_probs
