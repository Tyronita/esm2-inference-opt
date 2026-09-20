# =============================================================================
# TRACK B  alias: hf / hf-transformers / transformers-fp16
#   Same weights as Track A. Bridge: rho=+0.999973. See scoring/ref/wt_marginals.py for Track A.
# =============================================================================
"""
Wildtype marginal scoring.

TRACK B: transformers==4.44.0 HuggingFace API (fp16).
For the reference implementation used in benchmarks, see scoring/ref/wt_marginals.py.

Source: Meier et al. NeurIPS 2021 (ESM-1v paper), §Scoring Methods
ProteinGym: compute_fitness.py strategy="wt-marginals"

Score = log p(mut_aa | full_wt_sequence) - log p(wt_aa | full_wt_sequence)

Cost: 1 forward pass per assay (regardless of sequence length or variant count).

Limitation: Model sees position i when computing its probability — self-information
leaks into the score. Systematically overconfident vs masked_marginals.
For sequences > 1022 AA: windowed approach used (overlapping 1024-token windows).
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
) -> list[float]:
    model.eval()
    L = len(sequence)

    if L <= 1022:
        log_probs = _single_pass(model, tokenizer, sequence, device)
    else:
        log_probs = _windowed_pass(model, tokenizer, sequence, device)

    scores = []
    for variant_str in variants:
        score = 0.0
        for mut in variant_str.split(":"):
            wt_aa, pos_1, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
            tok_pos = pos_1  # 1-indexed protein pos = token pos (BOS at 0 offsets to 1)
            wt_id  = tokenizer.convert_tokens_to_ids(wt_aa)
            mut_id = tokenizer.convert_tokens_to_ids(mut_aa)
            score += (log_probs[tok_pos, mut_id] - log_probs[tok_pos, wt_id]).item()
        scores.append(score)
    return scores


def _single_pass(model, tokenizer, sequence, device):
    enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    logits = model(input_ids=enc["input_ids"]).logits[0]       # (L+2, vocab)
    return torch.log_softmax(logits.float(), dim=-1)            # (L+2, vocab)


def _windowed_pass(model, tokenizer, sequence, device, window=1022, overlap=256):
    """Overlapping sigmoid-weighted windows for sequences > 1022 AA.
    Matches ProteinGym compute_fitness.py 'overlapping' strategy."""
    L = len(sequence)
    vocab = model.config.vocab_size
    log_probs  = torch.full((L + 2, vocab), float("-inf"), device=device)
    weight_sum = torch.zeros(L + 2, device=device)

    start = 0
    while start < L:
        end = min(start + window, L)
        chunk = sequence[start:end]
        enc = tokenizer(chunk, return_tensors="pt", add_special_tokens=True).to(device)
        lp  = torch.log_softmax(
            model(input_ids=enc["input_ids"]).logits[0].float(), dim=-1
        )  # (chunk_len+2, vocab)

        # Sigmoid weights: upweight centre, downweight edges
        chunk_L = end - start
        positions = torch.arange(chunk_L, dtype=torch.float32, device=device)
        weights = torch.sigmoid((positions - overlap) / 10) * torch.sigmoid(
            (chunk_L - 1 - positions - overlap) / 10
        )
        weights = weights.clamp(min=1e-6)

        for local_i, global_i in enumerate(range(start + 1, end + 1)):  # skip BOS
            w = weights[local_i]
            prev = log_probs[global_i]
            if prev.isinf().all():
                log_probs[global_i] = lp[local_i + 1]
                weight_sum[global_i] = w
            else:
                # Weighted log-sum-exp merge
                log_probs[global_i] = torch.logaddexp(
                    prev + math.log(weight_sum[global_i].item()),
                    lp[local_i + 1] + math.log(w.item()),
                ) - math.log(weight_sum[global_i].item() + w.item())
                weight_sum[global_i] += w

        start += window - overlap

    return log_probs
