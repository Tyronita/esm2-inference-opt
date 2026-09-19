# Results Log

Append-only. Each entry is one run. Never edit existing entries.
Pull from Modal volume: `python -m modal run modal_app.py::pull_results`

---

## 2026-09-12T00:00:00Z  Mac M3 MPS — ESM-2 650M timing (measured)

```
hardware : Apple M3, MPS backend, torch 2.14.0, fp16
model    : facebook/esm2_t33_650M_UR50D  (651M params)
method   : masked_marginals (L forward passes)
```

| Protein | L  | Time (MPS) | Notes |
|---------|-----|------------|-------|
| SNCA    | 140 | 16.5 s     | full 140 passes |
| LRRK2   | 2527 | 85.8 min  | full 2527 passes |

L² scaling check: (2527/140)² = 326× expected, observed 312× ✓

Full ProteinGym (217 assays, mean L=397): ~500 hr estimated (not run).

---

## 2026-09-19T00:00:00Z  Mac M3 MPS — smoke_test.py 44/44 (ESM-2 8M proxy)

```
hardware : Apple M3, MPS backend, torch 2.14.0
model    : facebook/esm2_t6_8M_UR50D  (8M params, same arch as 650M)
note     : 10 SNCA variants, L=140, 10 unique positions
```

| Method                  | Time    | Notes                         |
|-------------------------|---------|-------------------------------|
| masked_marginals        | 2.06 s  | 10 passes (10 unique positions) |
| wt_marginals            | 0.19 s  | 1 pass                         |
| pseudo_ppl              | 0.06 s  | mutant-seq passes              |
| batched_masked B=8      | 0.01 s  | ceil(10/8)=2 passes            |
| batched_masked B=32     | 0.01 s  | ceil(10/32)=1 pass             |
| simple_ofs              | 0.01 s  | 1 pass                         |

Precision gate (fp16 vs fp32, same model):
- Spearman ρ = +1.0000  (gate ≥ 0.999) ✓
- Max |diff|  = 0.00922  (gate < 0.05)  ✓

All imports: torch, transformers 4.44.0, fair-esm @ 2b369911, biopython 1.79,
numpy 1.26.4, scipy 1.13.1, pandas 2.2.2, scikit-learn 1.5.1 — all PASS.

44/44 checks passed. ALL CLEAR — safe to run on Modal A100.

---

<!-- Modal runs will be appended below by pull_results -->
