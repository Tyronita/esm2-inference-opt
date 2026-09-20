# protein-language-model-experiments-esm2_t33_650M_UR50D

Early inference experiments with FAIR's ESM-2 650M protein language model.
Archived here as a subtree of [esm2-inference-opt](https://github.com/Tyronita/esm2-inference-opt).

## What was here

| File | Migrated to |
|---|---|
| `NUMBERS.md` (FLOPs, token costs, H100 benchmarks) | `docs/inference_cost_analysis.md` |
| `shed/oracle.py`, `shed/esm_arm.py`, `shed/budget.py` | superseded by `scoring/` |
| `pd_variants/pd_genes_clinvar.tsv` (13,581 ClinVar variants) | `LewyGym/pathogenicity/` |

The substantive content has been migrated. This directory preserves the original repo's
git history via `git subtree`.
