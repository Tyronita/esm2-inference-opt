#!/usr/bin/env python3
"""
Push benchmark results to HuggingFace Hub as a dataset.

Dataset: EvanOLeary/esm2-proteinGym-benchmark
Source:  results/ref_proteingym_stream.jsonl

Schema (one row per assay × method):
  assay_id, sequence_length, n_variants, method, implementation, dtype, gpu
  wall_mean, wall_std, wall_runs (list), warmup_wall_s (list)
  cuda_ms_mean, cuda_ms_std, peak_mem_mb
  n_warmup, n_timed
  spearman_rho, spearman_p, pearson_r, pearson_p
  ndcg_10, ndcg_50, top5_recall, top10_recall, auc, mcc, fraction_correct

Usage:
    huggingface-cli login          # or export HF_TOKEN=...
    python scripts/push_hf_dataset.py
    python scripts/push_hf_dataset.py --dry-run   # print first 3 rows, no push
    python scripts/push_hf_dataset.py --stream results/ref_proteingym_stream.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

REPO   = Path(__file__).parent.parent
HF_DATASET_ID = "EvanOLeary/esm2-proteinGym-benchmark"

DATASET_CARD = """\
---
license: mit
task_categories:
  - other
language:
  - en
tags:
  - biology
  - protein
  - benchmark
  - ESM-2
  - ProteinGym
  - variant-effect-prediction
pretty_name: ESM-2 650M ProteinGym Benchmark
---

# ESM-2 650M ProteinGym Benchmark

Reproduction of [Notin et al. NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html)
Table 1 (ESM-2 650M, 217 substitution assays) plus A100 wall-clock timing.

## Implementation

- **Model**: `facebook/esm2_t33_650M_UR50D` (650M parameters)
- **Implementation**: `fair-esm @ 2b369911` (fp32) — the original Meta Research reference
- **Hardware**: NVIDIA A100-SXM4-40GB (Modal)
- **Timing**: N=2 warmup passes discarded; N=5 timed passes per assay; mean ± std reported

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `assay_id` | string | ProteinGym DMS identifier |
| `method` | string | Scoring method (wt_marginals, masked_marginals, pseudo_ppl) |
| `spearman_rho` | float | Spearman ρ vs DMS fitness |
| `wall_mean` | float | Mean wall-clock time (s) over N timed runs |
| `wall_std` | float | Std dev of wall times |
| `wall_runs` | list[float] | Per-run wall times |
| `cuda_ms_mean` | float | Mean CUDA device time (ms, from torch.cuda.Event) |
| `peak_mem_mb` | float | Peak GPU memory (MB) |
| `implementation` | string | `fair-esm@2b369911` |

## Published targets (Notin et al. 2023)

| Method | Published ρ |
|--------|------------|
| wt_marginals | 0.430 |
| masked_marginals | 0.440 |
| pseudo_ppl | 0.440 |

## Repo

[esm2-inference-opt](https://github.com/Tyronita/esm2-inference-opt)
"""


def load_rows(stream_path: Path) -> list[dict]:
    rows = []
    with open(stream_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                # Ensure list fields are lists (not None)
                for k in ("wall_runs", "warmup_wall_s"):
                    if row.get(k) is None:
                        row[k] = []
                # Drop nested profile dict — too complex for HF dataset rows
                row.pop("profile", None)
                rows.append(row)
    return rows


def push(rows: list[dict], dry_run: bool = False):
    from datasets import Dataset

    ds = Dataset.from_list(rows)
    print(f"  Dataset: {len(ds)} rows × {len(ds.column_names)} columns")
    print(f"  Columns: {ds.column_names}")

    if dry_run:
        print("\n  [dry-run] first 3 rows:")
        for i in range(min(3, len(ds))):
            r = ds[i]
            print(f"    [{i}] {r['assay_id']}  {r['method']}  "
                  f"ρ={r['spearman_rho']:+.4f}  wall={r['wall_mean']:.2f}s")
        return

    print(f"\n  Pushing to {HF_DATASET_ID} ...")
    ds.push_to_hub(HF_DATASET_ID, token=True)

    # Write dataset card
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        commit_message="Update dataset card",
    )
    print(f"  Done → https://huggingface.co/datasets/{HF_DATASET_ID}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default=str(REPO / "results" / "ref_proteingym_stream.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stream_path = Path(args.stream)
    if not stream_path.exists():
        print(f"  Stream not found: {stream_path}")
        print("  Run the benchmark first:")
        print("    modal run --detach modal_app.py::ablate_ref_proteingym")
        print("    modal run modal_app.py::pull_ref_results")
        sys.exit(1)

    rows = load_rows(stream_path)
    print(f"  Loaded {len(rows)} rows from {stream_path}")

    if len(rows) == 0:
        print("  No rows — nothing to push.")
        sys.exit(0)

    # Print coverage summary
    from collections import Counter
    by_method = Counter(r["method"] for r in rows)
    print("  Coverage:")
    for m, n in sorted(by_method.items()):
        rhos = [r["spearman_rho"] for r in rows if r["method"] == m]
        mean_rho = sum(rhos) / len(rhos)
        print(f"    {m:<25}  {n:3d} assays  mean ρ={mean_rho:+.4f}")

    push(rows, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
