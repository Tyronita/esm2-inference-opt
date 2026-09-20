"""
Scoring method registry — Track C (ESMC 600M, EvolutionaryScale SDK).

Two methods:
  masked_marginals  — sequential reference (ESMC SDK, exact, L passes)
  batched_masked_32 — batched shinkaevolve (EsmcForMaskedLM, exact, ceil(L/32) passes)

Model loaded separately: ESMC.from_pretrained("esmc_600m") or EsmcForMaskedLM.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class ScoringMethod:
    name: str
    fn: Callable
    passes: str
    source: str
    published_rho: float


def get_registry_sdk() -> list[ScoringMethod]:
    """Registry for sequential ESMC SDK (load_model_esmc → ESMC object)."""
    from scoring.esmc import masked_marginals

    return [
        ScoringMethod(
            name="masked_marginals",
            fn=masked_marginals.score_variants,
            passes="L",
            source="ProteinGym evoscale/compute_fitness.py::_score_mutations_common",
            published_rho=0.0,
        ),
    ]


def get_registry_hf() -> list[ScoringMethod]:
    """Registry for batched EsmcForMaskedLM (load_model_esmc_hf → EsmcForMaskedLM object)."""
    from scoring.esmc import batched_masked

    return [
        ScoringMethod(
            name="batched_masked_32",
            fn=lambda m, t, s, v, d: batched_masked.score_variants(m, t, s, v, d, batch_size=32),
            passes="ceil(L/32)",
            source="shinkaevolve: true-batch EsmcForMaskedLM, B=32",
            published_rho=0.0,
        ),
        ScoringMethod(
            name="batched_masked_64",
            fn=lambda m, t, s, v, d: batched_masked.score_variants(m, t, s, v, d, batch_size=64),
            passes="ceil(L/64)",
            source="shinkaevolve: true-batch EsmcForMaskedLM, B=64",
            published_rho=0.0,
        ),
    ]
