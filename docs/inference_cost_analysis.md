# Inference cost — the comparable numbers

All figures computed from the actual reference files, not estimated.
Sources: `ProteinGym/reference_files/DMS_substitutions.csv`,
`evo2/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx` (Findlay 2018),
`evo2/notebooks/brca1/brca1_zero_shot_vep.ipynb` (WINDOW_SIZE = 8192).

FLOPs use the standard forward-pass approximation **2 × params × tokens**.
Wall-clock assumes one H100 at 30% MFU (~990 TFLOP/s bf16 dense peak). Real
throughput will be worse — treat these as floors, not predictions.

---

## The two benchmarks

| | ProteinGym (substitutions) | BRCA1 zero-shot VEP |
|---|---|---|
| Assays / variants | 217 assays, 2,465,767 variants | 3,893 SNVs |
| — singles | 696,311 | 3,893 |
| — multiples | 1,769,456 | 0 |
| Unique positions | — | 1,326 |
| Sequence length | mean 397, median 245, max 3,423 | fixed 8,192 window |
| Scoring | masked marginal | ref + variant likelihood delta |
| Metric | Spearman per assay, aggregated | AUROC vs LOF label |

**ProteinGym token cost: 89M token-positions.**
Sum of L² across 217 assays. Mask each position once, score all variants at
that position for free.

**BRCA1 token cost: 42.8M tokens.**
3,893 variant windows (31.9M) + 1,326 deduplicated reference windows (10.9M),
all at 8,192.

Scoring ProteinGym naively — one forward pass per variant instead of per
position — costs **972M tokens, 11× more**. Don't.

---

## Cost by model

| Arm | Params | Tokens | FLOPs | H100 @30% MFU |
|---|---|---|---|---|
| Evo 2 7B / BRCA1 | 7B | 42.8M | 599 PF | ~34 min |
| Evo 2 1B / BRCA1 | 1B | 42.8M | 85.5 PF | ~4.8 min |
| HyenaDNA 6.6M / BRCA1 | 6.6M | 42.8M | 0.56 PF | ~2 s |
| ESM-2 650M / ProteinGym full | 650M | 88.7M | 115 PF | ~6.5 min |
| ESM-2 650M / PG-lite (20 assays) | 650M | 8.0M | 10.4 PF | ~35 s |
| ESM-2 8M / ProteinGym full | 8M | 88.7M | 1.4 PF | ~5 s |

---

## What this means for Saturday

**The full benchmarks are cheap.** Both canonical evals fit in under an hour of
single-GPU time even at 7B. You are not compute-limited on evaluation — you are
limited by how long it takes to get the code right. Budget accordingly: the
oracle is an engineering problem, not a compute problem.

**HyenaDNA is effectively free.** Two seconds for the full BRCA1 eval. There is
no reason to develop the harness against anything else. Every debug cycle on
Evo 2 7B costs you 34 minutes; on HyenaDNA it costs you the time to press enter.
Get the pipeline green on the 6.6M model, then run 7B once.

**ESM-2 8M is your smoke test.** Five seconds for the full 217-assay benchmark.
Correctness of the scoring code is independent of model size, so validate the
masked-marginal implementation at 8M, then swap the checkpoint.

**Where the cost actually is.** Five ProteinGym assays are ~47% of the total,
because masked-marginal cost scales as L²:

| Assay | L | tokens |
|---|---|---|
| A0A140D2T1_ZIKV_Sourisseau_2019 | 3,423 | 12M |
| BRCA2_HUMAN_Erwood_2022_HEK293T | 3,418 | 12M |
| POLG_HCVJF_Qi_2014 | 3,033 | 9M |
| POLG_CXB3N_Mattenberger_2021 | 2,185 | 5M |
| SCN5A_HUMAN_Glazer_2019 | 2,016 | 4M |

Drop those five and the remaining 212 cost about half.

**Lite-arm warning.** The 20 *cheapest* assays average L=45 — short peptides,
totally unrepresentative. Your Spearman won't resemble 0.414 and you'll burn an
hour chasing a gap that isn't a bug. Stratify by sequence length instead.

---

## Reference scores to hit

| Model | Benchmark | Target |
|---|---|---|
| ESM-2 650M | ProteinGym substitutions | 0.414 ± 0.012 Spearman |
| ESM-2 3B | same | 0.406 |
| ESM-2 15B | same | 0.401 |
| ESM-2 650M | UniRef50 held-out | ~6.95 perplexity (7.00 on BioNeMo's split) |

Per-assay published scores are in
`ProteinGym/benchmarks/DMS_zero_shot/substitutions/Spearman/` — validate
assay-by-assay, not against the aggregate. An aggregate that matches can still
hide two errors cancelling.

---

## Caveats

- 2 × params × tokens ignores attention's L² term. At L=8,192 on Evo 2 that
  understates the attention layers; at L≈400 on ESM-2 it's close enough.
- 30% MFU is optimistic for unoptimised inference on short sequences. Expect
  2–4× worse until batching is right.
- Evo 2 7B in bf16 runs without Transformer Engine. 20B and 40B need TE + FP8 +
  Hopper.
