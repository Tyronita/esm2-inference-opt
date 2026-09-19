"""
All known scoring methods for ESM-2 masked language models on ProteinGym.

Methods and their sources:
  wt_marginals     Meier et al. NeurIPS 2021 (ESM-1v paper), §Methods
  masked_marginals Meier et al. NeurIPS 2021 — canonical ProteinGym baseline
  pseudo_ppl       Meier et al. NeurIPS 2021 / ProteinGym compute_fitness.py
  batched_masked   Generalization of masked_marginals (B positions per pass)
  simple_ofs       Naive single-pass approximation — NOT the full Kantroo 2024
                   OFS (which requires separately trained MLP heads)

Limitations per method are documented in LIMITATIONS.md.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class ScoringMethod:
    name: str
    fn: Callable
    passes: str        # "1", "L", "L/B", "L (mutant)"
    source: str
    note: str


def get_registry(device: str) -> list[ScoringMethod]:
    from scoring import batched_masked, masked_marginals, pseudo_ppl, simple_ofs, wt_marginals

    return [
        ScoringMethod(
            name="wt_marginals",
            fn=wt_marginals.score_variants,
            passes="1",
            source="Meier et al. NeurIPS 2021",
            note="1 forward pass; model sees position being scored (self-info leakage)",
        ),
        ScoringMethod(
            name="masked_marginals",
            fn=masked_marginals.score_variants,
            passes="L",
            source="Meier et al. NeurIPS 2021 (canonical ProteinGym baseline)",
            note="L passes; position isolated from context. Published ρ=0.44 on ESM-2 650M",
        ),
        ScoringMethod(
            name="pseudo_ppl",
            fn=pseudo_ppl.score_variants,
            passes="L (mutant)",
            source="Meier et al. NeurIPS 2021 / ProteinGym compute_fitness.py",
            note="L passes on mutant sequence; identical to masked_marginals for single-site variants",
        ),
        ScoringMethod(
            name="batched_masked_b8",
            fn=lambda m, t, s, v, d: batched_masked.score_variants(m, t, s, v, d, batch_size=8),
            passes="ceil(L/8)",
            source="This work",
            note="Approximation: 8 positions masked simultaneously. Attention patterns differ from single-mask",
        ),
        ScoringMethod(
            name="batched_masked_b32",
            fn=lambda m, t, s, v, d: batched_masked.score_variants(m, t, s, v, d, batch_size=32),
            passes="ceil(L/32)",
            source="This work",
            note="Approximation: 32 positions masked simultaneously. Faster but larger approximation error",
        ),
        ScoringMethod(
            name="simple_ofs",
            fn=simple_ofs.score_variants,
            passes="1",
            source="Naive approximation (NOT Kantroo et al. 2024)",
            note="WARNING: functionally equivalent to wt_marginals. True OFS requires trained MLP heads "
                 "(Kantroo, Wagner & Machta 2024, arXiv:2407.07265) not included here.",
        ),
    ]
