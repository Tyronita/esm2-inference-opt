# ESM-2 650M Inference Optimisation

Reproducing and beating the published ProteinGym baseline for ESM-2 650M,
running on Modal (A100/T4 on-demand).

## The target

> "ESM-2 650M achieves a mean Spearman ρ of **0.44** across 217 DMS substitution
> assays on ProteinGym using masked marginal log-likelihood ratio scoring."
>
> — Notin et al. (2023), *ProteinGym: Large-Scale Benchmarks for Protein Fitness
> Prediction and Design*, NeurIPS 2023

> "Masked marginal scoring requires **L independent forward passes** for a sequence
> of length L — one per position — making it L× more expensive than a single
> inference call."
>
> — Meier et al. (2021), *Language models enable zero-shot prediction of the effects
> of mutations on protein function*, NeurIPS 2021

> "The One Fell Swoop (OFS) approach makes do with just a **single forward pass**
> through the encoder model... ESM2 OFS pseudo-perplexity performs **nearly as well
> as the true pseudo-perplexity** at fitness estimation and defines a new state of
> the art on the ProteinGym Indels benchmark."
>
> — Kantroo, Wagner & Machta (2024), *Pseudo-perplexity in One Fell Swoop for
> Protein Fitness Estimation*, PRX Life / arXiv:2407.07265

## The gap we're closing

| Method | Forward passes | Spearman ρ (217 DMS) |
|---|---|---|
| Wildtype marginal | 1 | ~0.40 |
| **ESM-2 650M masked marginal** | **L** | **0.44 (baseline)** |
| OFS pseudo-perplexity | 1 | ~0.44 (matches) |
| ESM-1v ensemble (5 models) | 5L | ~0.47 |

**Goal:** reproduce the 0.44 baseline, then match it in 1 pass (OFS), then push higher.

## Stack

- **Model:** `facebook/esm2_t33_650M_UR50D` (650M params, 33 layers, d=1280, h=20)
- **Compute:** Modal — pure reproducible environment, A100 on demand
- **Benchmark:** ProteinGym substitution benchmark (217 DMS assays)
- **PD focus:** SNCA (A53T, E46K, A30P), LRRK2 (G2019S), GBA (N370S, L444P)

## Scoring methods

| File | Method | Cost | Notes |
|---|---|---|---|
| `scoring/masked_marginal.py` | Masked marginal | L passes | Canonical baseline |
| `scoring/ofs.py` | OFS pseudo-perplexity | 1 pass | Kantroo et al. 2024 |
| `scoring/wt_marginal.py` | Wildtype marginal | 1 pass | Fast lower bound |

## Quick start

```bash
# authenticate once
modal setup

# run baseline on PD proteins (fast, ~5 min on T4)
modal run modal_app.py::score_pd_proteins

# run full ProteinGym sweep (217 assays, ~2-3hr on A100)
modal run modal_app.py::score_proteingym
```

## References

- Lin et al. (2023). Evolutionary-scale prediction of atomic-level protein structure
  with a language model. *Science*, 379(6637), 1123–1130.
- Notin et al. (2023). ProteinGym: Large-Scale Benchmarks for Protein Fitness
  Prediction and Design. *NeurIPS 2023*.
- Meier et al. (2021). Language models enable zero-shot prediction of the effects
  of mutations on protein function. *NeurIPS 2021*.
- Kantroo, Wagner & Machta (2024). Pseudo-perplexity in One Fell Swoop for Protein
  Fitness Estimation. *PRX Life*. arXiv:2407.07265.
