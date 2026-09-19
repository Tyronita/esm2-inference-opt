# ESM-2 650M Inference Optimisation

Reproducing and beating the published ProteinGym baseline for ESM-2 650M,
running on Modal (A100 on-demand). All implementations pinned to exact git SHAs
so every number is reproducible.

## The target

> "ESM-2 650M achieves a mean Spearman ρ of **0.414 ± 0.012** across 217 DMS
> substitution assays on ProteinGym using masked marginal log-likelihood ratio scoring."
>
> — Notin et al. (2023), *ProteinGym: Large-Scale Benchmarks for Protein Fitness
> Prediction and Design*, NeurIPS 2023

## Scoring methods

| Module | Method | Passes/assay | Expected ρ | Notes |
|---|---|---|---|---|
| `scoring/wt_marginals.py` | Wildtype marginal | 1 | ~0.40 | 1 pass, no masking |
| `scoring/masked_marginals.py` | Masked marginal | L | **0.414** | Published baseline |
| `scoring/pseudo_ppl.py` | Pseudo-perplexity | L | ~0.414 | = masked_marginals for singles |
| `scoring/batched_masked.py` | Batched masked B=8 | ceil(L/8) | ~0.413 | ~8× faster, small error |
| `scoring/batched_masked.py` | Batched masked B=32 | ceil(L/32) | ~0.410 | ~32× faster, larger error |
| `scoring/simple_ofs.py` | Simple 1-pass | 1 | ~0.40 | ≡ wt_marginals (not true OFS) |

True OFS (Kantroo et al. 2024) requires trained MLP heads — not implemented here.

---

## What we did

1. Created repo with `refs/` git submodules pinned to exact SHAs
2. Pinned fair-esm @ `2b369911` — the exact commit that produced ρ=0.414
3. Pinned ProteinGym @ `144fe22b` — source of the 217-assay metadata
4. Implemented 6 scoring methods in `scoring/`
5. Implemented 10 metrics in `benchmark/metrics.py` (Spearman, Pearson, NDCG, AUC, MCC, top-k recall…)
6. Corrected baseline from 0.44 → **0.414 ± 0.012** (from published table in `NUMBERS.md`)
7. Wrote `refs/graph.yml` — full dependency graph as (subject, predicate, object) triples
8. Wrote `check_consistency.py` — compares fair-esm fp32 vs transformers fp16 on SNCA, gate ρ ≥ 0.999
9. Built Modal image — baked once (90s), cached forever (`im-tPKQcOYJ2kBgvntr2EiOq7`)
10. Ran `smoke_test.py` on M3 MPS — **44/44 checks pass**, fp16 vs fp32 ρ=1.000

---

## Why Modal, not local

We are doing **inference-time benchmarking** — the metric is how fast we can score
217 assays against a known ρ=0.414 baseline. That makes hardware reproducibility
a first-class requirement, not a convenience.

| Factor | M3 MPS | A100 (Modal) |
|---|---|---|
| SNCA (L=140) | 16.5 s | ~0.3 s |
| LRRK2 (L=2527) | **85.8 min** | ~1.7 min |
| Full 217-assay run | ~500 hr | **~23 min** |
| fp16 support | partial (MPS gaps) | full |
| Reproducible env | no (system libs) | yes (baked image) |
| Cost | "free" but unusable | ~$1/run |

MPS cannot run full ProteinGym in a session — LRRK2 alone is 86 minutes and it
scales as L². Modal gives the same GPU every run, so timing numbers are comparable
across experiments. The baked image means no install overhead per run (~10s cold
start vs ~90s first build, then cached forever).

---

## Mac M3 MPS timing (measured)

All figures from actual runs, not estimates.

### ESM-2 650M (the model we benchmark)

| Protein | L | Method | M3 MPS time | A100 equiv (~50×) |
|---|---|---|---|---|
| SNCA (α-synuclein) | 140 | masked_marginals | **16.5 s** | ~0.3 s |
| LRRK2 | 2527 | masked_marginals | **85.8 min** | ~1.7 min |
| Full ProteinGym (217 assays) | mean L=397 | masked_marginals | ~500 hr (est.) | ~23 min |

L² scaling confirmed: LRRK2/SNCA ratio = (2527/140)² = 326×. Observed = 85.8min/16.5s = 312× ✓

### ESM-2 8M (smoke test proxy — same code path, 30MB)

From `smoke_test.py` on M3 MPS:

| Method | Time (10 variants, L=140) | Notes |
|---|---|---|
| masked_marginals | 2.06 s | 10 passes (one per unique position) |
| wt_marginals | 0.19 s | 1 pass |
| pseudo_ppl | 0.06 s | mutant-seq passes |
| batched_masked B=8 | 0.01 s | ceil(10/8)=2 passes |
| batched_masked B=32 | 0.01 s | ceil(10/32)=1 pass |
| simple_ofs | 0.01 s | 1 pass |

**fp16 vs fp32 consistency (8M, MPS): ρ = +1.0000, max |diff| = 0.009** — precision is not an issue.

---

## All ESM-2 model configs (from paper + ProteinGym benchmark)

Architecture from `refs/fair-esm/esm/model/esm2.py`: FFN = 4 × d, RoPE positional encoding,
pre-LayerNorm, GELU. No positional embeddings beyond RoPE. Max input = 1024 tokens (1022 AA + BOS/EOS).

| Model ID | Layers | d | Heads | FFN | Params | ProteinGym ρ | Rank |
|---|---|---|---|---|---|---|---|
| esm2_t6_8M_UR50D | 6 | 320 | 20 | 1280 | 8M | 0.226 ± 0.015 | 93 |
| esm2_t12_35M_UR50D | 12 | 480 | 20 | 1920 | 35M | 0.321 ± 0.015 | 84 |
| esm2_t30_150M_UR50D | 30 | 640 | 20 | 2560 | 150M | 0.387 ± 0.013 | 62 |
| **esm2_t33_650M_UR50D** | **33** | **1280** | **20** | **5120** | **650M** | **0.414 ± 0.012** | **45** |
| esm2_t36_3B_UR50D | 36 | 2560 | 40 | 10240 | 3B | 0.406 ± 0.011 | 50 |
| esm2_t48_15B_UR50D | 48 | 5120 | 40 | 20480 | 15B | 0.400 ± 0.010 | 52 |

Source: `refs/ProteinGym/benchmarks/.../Summary_performance_DMS_substitutions_Spearman.csv`
Architecture: `refs/fair-esm/esm/model/esm2.py:14`

**Key observation from paper**: 650M is the sweet spot. 3B and 15B are *worse* on ProteinGym
despite 5–23× more parameters. The scaling law doesn't hold for variant effect prediction —
larger models overfit to evolutionary structure that doesn't transfer to fitness.

---

## Empirical timing model

`t(L) = β × L²` — total cost of masked_marginals on a sequence of length L.
L passes (one per position) × O(L) tokens per pass = O(L²).

| Hardware | β (s/(AA)²) | Source |
|---|---|---|
| M3 MPS (measured) | 8.25 × 10⁻⁴ | SNCA + LRRK2 timing |
| A100 40GB (expected) | ~1.6 × 10⁻⁵ | ~50× MPS speedup |

β̂ is fit from live data during every Modal run — `RunTimer` in `benchmark/timing.py`
maintains a rolling median β̂ and prints ETA after each assay.

---

## Running

```bash
# local smoke test (M3 / any machine, no GPU spend)
source .venv/bin/activate
python smoke_test.py   # should print: 44/44 passed  ALL CLEAR

# cross-library consistency check (~$0.30, A100)
python -m modal run modal_app.py::check_consistency

# PD proteins ablation — 3 assays × 6 methods (~$0.90, A100, ~15 min)
python -m modal run modal_app.py::ablate_pd_proteins

# Full ProteinGym — 217 assays × 6 methods (~$12, A100, ~3-4 hr)
python -m modal run modal_app.py::ablate_proteingym
```

---

## Stack

- **Model:** `facebook/esm2_t33_650M_UR50D` (650M params, 33 layers, d=1280, h=20)
- **Compute:** Modal — A100-40GB on demand, baked image, persistent volume for weights
- **Benchmark:** ProteinGym substitution benchmark v1.1 (217 DMS assays)
- **Reference impl:** `refs/fair-esm/` @ `2b369911` (the commit that produced ρ=0.414)
- **PD focus:** SNCA (A53T, E46K, A30P), LRRK2 (G2019S), GBA (N370S, L444P)

---

## References

- Lin et al. (2023). Evolutionary-scale prediction of atomic-level protein structure
  with a language model. *Science* 379(6637). doi:10.1126/science.ade2574
- Notin et al. (2023). ProteinGym: Large-Scale Benchmarks for Protein Fitness
  Prediction and Design. *NeurIPS 2023*.
  proceedings.neurips.cc/paper_files/paper/2023/file/cac723e5ff29f65e3fcbb0739ae91bee
- Meier et al. (2021). Language models enable zero-shot prediction of the effects
  of mutations on protein function. *NeurIPS 2021*. bioRxiv 10.1101/2021.07.09.450648
- Kantroo, Wagner & Machta (2024). Pseudo-perplexity in One Fell Swoop for Protein
  Fitness Estimation. *PRX Life*. arXiv:2407.07265
