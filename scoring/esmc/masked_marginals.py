"""
Masked marginal scoring — Track C sequential (ESMC 600M, EvolutionaryScale SDK).

Algorithm source:
  refs/ProteinGym/proteingym/baselines/evoscale/compute_fitness.py::_score_mutations_common

Model API from:
  refs/evolutionaryscale-esm/esm/models/esmc/  (ESMC SDK)

Score = Σ_i∈mut [ log p(x*_i | x_{-i}) − log p(x_i | x_{-i}) ]

Cost: |unique positions| forward passes per assay (sequential, exact).
Long sequences (L > 1022): optimal window centred on mutation position.
"""

import re

import attr
import torch
from esm.sdk.api import ESMProtein, LogitsConfig

_AA = "ACDEFGHIKLMNPQRSTVWY"
_MASK_TOKEN_ID = 32   # <mask>; refs/evolutionaryscale-esm/esm/models/esmc/tokenizer.py:45
_WINDOW = 1022        # matches ProteinGym evoscale default (window_size=1024 - 2 special tokens)


@torch.no_grad()
def score_variants(
    model,
    tokenizer,          # unused — ESMC SDK encodes via model.encode()
    sequence: str,
    variants: list[str],
    device: str = "cuda",
) -> list[float]:
    model.eval()

    # aa → token ID map via model's own encoder (mirrors compute_fitness.py lines 315-319)
    aa_to_token: dict[str, int] = {}
    for aa in _AA:
        t = model.encode(ESMProtein(sequence=aa))
        aa_to_token[aa] = int(t.sequence[1].item())  # skip BOS at index 0

    # Encode full wildtype sequence
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    seq_tokens = protein_tensor.sequence   # shape: (L+2,) — BOS + AAs + EOS
    L = len(sequence)
    needs_window = L > _WINDOW

    # Parse variants, collect unique 1-indexed positions
    parsed: list[list[tuple[str, int, str]]] = []
    for v in variants:
        muts: list[tuple[str, int, str]] = []
        for m in v.split(":"):
            match = re.match(r"([A-Z])(\d+)([A-Z])", m)
            if match:
                wt, pos_str, mt = match.groups()
                muts.append((wt, int(pos_str), mt))
        parsed.append(muts)

    positions_needed = sorted({pos for muts in parsed for _, pos, _ in muts})

    # One masked forward pass per unique position (compute_fitness.py lines 359-425)
    log_probs: dict[int, torch.Tensor] = {}
    for pos_1 in positions_needed:
        seq_pos = pos_1 - 1       # 0-indexed in sequence string
        token_pos = pos_1         # 1-indexed in token sequence (BOS at 0)

        if needs_window:
            half = _WINDOW // 2
            start = max(0, seq_pos - half)
            end = min(L, start + _WINDOW)
            if end == L:
                start = max(0, L - _WINDOW)
            window_seq = sequence[start:end]
            w_tensor = model.encode(ESMProtein(sequence=window_seq))
            local_pos = seq_pos - start + 1   # +1 for BOS
            masked = w_tensor.sequence.clone()
            masked[local_pos] = _MASK_TOKEN_ID
            m_tensor = attr.evolve(w_tensor, sequence=masked)
            out = model.logits(m_tensor, LogitsConfig(sequence=True))
            logits = out.logits.sequence[0, local_pos]
        else:
            masked = seq_tokens.clone()
            masked[token_pos] = _MASK_TOKEN_ID
            m_tensor = attr.evolve(protein_tensor, sequence=masked)
            out = model.logits(m_tensor, LogitsConfig(sequence=True))
            logits = out.logits.sequence[0, token_pos]

        log_probs[pos_1] = torch.log_softmax(logits, dim=-1)

    scores: list[float] = []
    for muts in parsed:
        score = 0.0
        for wt, pos_1, mt in muts:
            lp = log_probs[pos_1]
            score += (lp[aa_to_token[mt]] - lp[aa_to_token[wt]]).item()
        scores.append(score)

    return scores
