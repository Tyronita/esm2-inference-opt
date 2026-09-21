#!/usr/bin/env python3
"""
benchmark/proteingym_mlx.py  —  ProteinGym DMS substitution benchmark for ESMC-300M MLX.

Algorithm: batched masked-marginals
  score(mut) = log P(mut_aa | ctx_masked) - log P(wt_aa | ctx_masked)

Usage:
    .venv-esmc/bin/python3 benchmark/proteingym_mlx.py --opt-level 3 --max-assays 12

Outputs:
    results/proteingym_opt{N}.json   — per-assay results + summary
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from scipy.stats import spearmanr

from esm.models.esmc import EsmcTokenizer
from esm.models.esmc.config import ESMC_300M_HF_REPO
from scoring.esmc.mlx_esmc import EsmcMLX

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

REF_CSV = ROOT / "refs" / "ProteinGym" / "reference_files" / "DMS_substitutions.csv"
DMS_DIR = ROOT / "refs" / "ProteinGym" / "DMS_substitutions"

# Published ESMC-300M ProteinGym zero-shot performance (masked_marginals, 217 assays)
PUBLISHED_RHO = 0.406
PUBLISHED_SEM = 0.013


# ---------------------------------------------------------------------------
# Mutation parsing
# ---------------------------------------------------------------------------

_MUT_RE = re.compile(r"([A-Z])(\d+)([A-Z])")


def parse_mutation_string(mut_str: str) -> list[tuple[int, str, str]]:
    """Parse 'A23V' or 'A23V:G45L' into [(22, 'A', 'V'), (44, 'G', 'L')]."""
    muts = []
    for part in mut_str.split(":"):
        m = _MUT_RE.fullmatch(part.strip())
        if m:
            wt, pos_s, mut_aa = m.groups()
            muts.append((int(pos_s) - 1, wt, mut_aa))  # convert to 0-indexed
    return muts


# ---------------------------------------------------------------------------
# Per-assay scoring using EsmcMLX.score_variants_batched
# ---------------------------------------------------------------------------

def score_assay(
    model: EsmcMLX,
    tokenizer,
    sequence: str,
    variant_strings: list[str],
    batch_size: int = 12,
) -> list[float]:
    """Score all variants in one assay using batched masked-marginals."""
    mutations_per_variant = [parse_mutation_string(v) for v in variant_strings]

    # Flatten to (pos, wt, mut) tuples — one entry per single mutation
    # For multi-mutants: score is sum of individual masked-marginals
    # Collect unique positions
    all_mutations_flat: list[tuple[int, str, str]] = []
    for muts in mutations_per_variant:
        all_mutations_flat.extend(muts)

    # Deduplicate by position
    unique_pos_muts: dict[int, tuple[str, str]] = {}
    for pos, wt, mut in all_mutations_flat:
        if pos not in unique_pos_muts:
            unique_pos_muts[pos] = (wt, mut)

    # Build list of (pos, wt, mut) for unique positions — only unique positions,
    # keeping first seen wt/mut (we need the log-prob array anyway)
    unique_mutations = [(pos, wt, mut) for pos, (wt, mut) in sorted(unique_pos_muts.items())]

    # Use model.score_variants_batched — it deduplicates positions internally
    # and returns log P(mut|ctx) - log P(wt|ctx) per mutation
    scores_per_unique = model.score_variants_batched(
        tokenizer, sequence, unique_mutations, batch_size=batch_size
    )
    # Build a map: pos -> log_prob_array (we need raw log-probs, not differences)
    # score_variants_batched returns differences; we need to recompute per-variant
    # So instead call the internals directly or use the same approach but per variant.
    # Actually score_variants_batched already returns the delta per (pos,wt,mut) tuple.
    # For multi-mutants we need to sum over all mutations in the variant.
    pos_to_delta: dict[tuple[int, str, str], float] = {}
    for i, (pos, wt, mut) in enumerate(unique_mutations):
        pos_to_delta[(pos, wt, mut)] = scores_per_unique[i]

    # Now compute per-variant scores
    # For single-mutants: score = delta[(pos, wt, mut)]
    # For multi-mutants: score = sum of deltas (independence assumption, same as ProteinGym)
    variant_scores = []
    for muts in mutations_per_variant:
        if not muts:
            variant_scores.append(float("nan"))
            continue
        total = 0.0
        valid = True
        for pos, wt, mut in muts:
            key = (pos, wt, mut)
            if key in pos_to_delta:
                total += pos_to_delta[key]
            else:
                # Position not in unique map (shouldn't happen) — try to recompute
                valid = False
                break
        variant_scores.append(total if valid else float("nan"))

    return variant_scores


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    opt_level: int,
    max_assays: int | None,
    batch_size: int,
    assay_ids: list[str] | None = None,
) -> dict:
    t_start = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"ProteinGym DMS Benchmark  |  ESMC-300M MLX  |  opt={opt_level}")
    print(f"{'='*60}")

    # Load reference
    if not REF_CSV.exists():
        print(f"ERROR: reference CSV not found at {REF_CSV}")
        sys.exit(1)

    with open(REF_CSV) as f:
        ref_rows = list(csv.DictReader(f))

    # Filter to available DMS files
    available = []
    for r in ref_rows:
        p = DMS_DIR / r["DMS_filename"]
        if p.exists():
            available.append(r)

    print(f"Reference: {len(ref_rows)} assays total, {len(available)} DMS files found")

    # Select assay subset
    if assay_ids:
        selected = [r for r in available if r["DMS_id"] in assay_ids]
    else:
        # Sort by sequence length (shortest first = fastest)
        available.sort(key=lambda r: int(r["seq_len"]))
        selected = available[:max_assays] if max_assays else available

    print(f"Running {len(selected)} assays (sorted by L)\n")

    # Load model
    print(f"Loading ESMC-300M MLX (opt={opt_level})...")
    t_load = time.perf_counter()
    model = EsmcMLX.from_pretrained("biohub/ESMC-300M", opt_level=opt_level)
    tokenizer = EsmcTokenizer.from_pretrained(ESMC_300M_HF_REPO)
    load_s = time.perf_counter() - t_load
    print(f"  Loaded in {load_s:.1f}s\n")

    # Run assays
    per_assay = []
    all_rho = []

    header = f"{'DMS_id':<45} {'L':>5} {'N':>6} {'rho':>7} {'p':>8} {'time_s':>7}"
    print(header)
    print("-" * len(header))

    for r in selected:
        assay_id = r["DMS_id"]
        sequence = r["target_seq"]
        L = int(r["seq_len"])
        dms_path = DMS_DIR / r["DMS_filename"]

        with open(dms_path) as f:
            dms_rows = list(csv.DictReader(f))

        variants = [row["mutant"] for row in dms_rows]
        fitness = [float(row["DMS_score"]) for row in dms_rows]
        N = len(variants)

        t0 = time.perf_counter()
        try:
            scores = score_assay(model, tokenizer, sequence, variants, batch_size=batch_size)
        except Exception as e:
            print(f"  {assay_id:<45} ERROR: {e}")
            continue
        wall_s = time.perf_counter() - t0

        # Filter NaN pairs
        valid_pairs = [(s, f) for s, f in zip(scores, fitness) if not (
            s != s or f != f  # NaN check
        )]
        if len(valid_pairs) < 10:
            print(f"  {assay_id:<45} SKIP: too few valid ({len(valid_pairs)})")
            continue

        sc_v, ft_v = zip(*valid_pairs)
        rho, pval = spearmanr(sc_v, ft_v)
        all_rho.append(rho)

        row_out = {
            "assay_id": assay_id,
            "L": L,
            "N": N,
            "n_valid": len(valid_pairs),
            "spearman_rho": round(float(rho), 4),
            "pval": float(pval),
            "time_s": round(wall_s, 2),
            "variants_per_s": round(N / wall_s, 1) if wall_s > 0 else 0,
        }
        per_assay.append(row_out)

        print(f"  {assay_id:<45} {L:>5} {N:>6} {rho:>+7.3f} {pval:>8.2e} {wall_s:>7.1f}s")

    # Summary
    total_s = time.perf_counter() - t_start
    mean_rho = float(np.mean(all_rho)) if all_rho else float("nan")
    sem_rho = float(np.std(all_rho) / np.sqrt(len(all_rho))) if len(all_rho) > 1 else float("nan")

    print(f"\n{'='*60}")
    print(f"  Assays completed : {len(per_assay)}")
    print(f"  Mean Spearman ρ  : {mean_rho:+.4f} ± {sem_rho:.4f}")
    print(f"  Published target : {PUBLISHED_RHO:.3f} ± {PUBLISHED_SEM:.3f}  (ESM-C 300M, 217 assays)")
    print(f"  Total time       : {total_s:.1f}s  (model load: {load_s:.1f}s)")
    print(f"{'='*60}\n")

    result = {
        "opt_level": opt_level,
        "opt_name": EsmcMLX.OPT_NAMES.get(opt_level, "unknown"),
        "n_assays": len(per_assay),
        "mean_spearman_rho": round(mean_rho, 4),
        "sem_spearman_rho": round(sem_rho, 4),
        "published_rho": PUBLISHED_RHO,
        "published_sem": PUBLISHED_SEM,
        "model_load_s": round(load_s, 1),
        "total_s": round(total_s, 1),
        "per_assay": per_assay,
    }

    out_path = RESULTS / f"proteingym_opt{opt_level}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ProteinGym DMS benchmark for ESMC-300M MLX")
    parser.add_argument("--opt-level", type=int, default=3,
                        help="Optimization level (0-10, default 3)")
    parser.add_argument("--max-assays", type=int, default=12,
                        help="Max number of assays to run (sorted by L, default 12)")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="Batch size for masked-marginals (default 12)")
    parser.add_argument("--assay-ids", type=str, default=None,
                        help="Comma-separated list of assay IDs to run (overrides --max-assays)")
    args = parser.parse_args()

    assay_ids = None
    if args.assay_ids:
        assay_ids = [a.strip() for a in args.assay_ids.split(",")]

    run_benchmark(
        opt_level=args.opt_level,
        max_assays=args.max_assays,
        batch_size=args.batch_size,
        assay_ids=assay_ids,
    )


if __name__ == "__main__":
    main()
