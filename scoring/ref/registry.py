"""
Scoring method registry — Track A (refs/fair-esm @ 2b369911, fp32).

All methods use signature: score_variants(model, alphabet, sequence, variants, device)
where model, alphabet = fair_esm.pretrained.esm2_t33_650M_UR50D()

These are the three methods benchmarked in Notin et al. NeurIPS 2023 (ProteinGym).
Target ρ values (217 substitution assays, ESM-2 650M, from Table 1):
  wt_marginals     0.430
  masked_marginals 0.440
  pseudo_ppl       0.440  (≈ masked_marginals for single-site assays)
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class ScoringMethod:
    name: str
    fn: Callable
    passes: str
    source: str
    published_rho: float   # Notin et al. 2023 Table 1, ESM-2 650M, 217 assays


def get_registry() -> list[ScoringMethod]:
    from scoring.ref import masked_marginals, pseudo_ppl, wt_marginals

    return [
        ScoringMethod(
            name="wt_marginals",
            fn=wt_marginals.score_variants,
            passes="1",
            source="Meier et al. NeurIPS 2021",
            published_rho=0.430,
        ),
        ScoringMethod(
            name="masked_marginals",
            fn=masked_marginals.score_variants,
            passes="L",
            source="Meier et al. NeurIPS 2021 (canonical ProteinGym baseline)",
            published_rho=0.440,
        ),
        ScoringMethod(
            name="pseudo_ppl",
            fn=pseudo_ppl.score_variants,
            passes="L (mutant)",
            source="Meier et al. NeurIPS 2021 / ProteinGym compute_fitness.py",
            published_rho=0.440,
        ),
    ]
