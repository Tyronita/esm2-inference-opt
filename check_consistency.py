"""
Consistency check: score SNCA variants with BOTH implementations and compare.

Purpose: confirm our transformers/fp16 implementation reproduces the
fair-esm/fp32 scores that produced the published ρ=0.414.

Requirements:
    pip install fair-esm                  # Meta's original library
    pip install transformers torch scipy  # already in requirements

Usage:
    python check_consistency.py [--n 200] [--device cpu|cuda|mps]

Output:
    - Spearman correlation between fair-esm scores and our scores
    - Max absolute difference
    - Mean absolute difference
    - Per-variant scatter (top 10 outliers)

If Spearman ρ(fair-esm, ours) ≥ 0.999, the implementations are consistent.
If it's < 0.999, investigate the outliers.

References:
    fair-esm:    refs/fair-esm/  (submodule)
    ProteinGym:  refs/ProteinGym/ (submodule)
    Original scoring script:
        refs/ProteinGym/proteingym/baselines/esm/compute_fitness.py
    get_optimal_window:
        refs/ProteinGym/proteingym/utils/scoring_utils.py:1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

# ── Data ─────────────────────────────────────────────────────────────────────
# SNCA (α-synuclein, UniProt P37840) — L=140, well-characterised PD gene
# Sequence from ProteinGym reference_files/DMS_substitutions.csv
# Assay: SYUA_HUMAN_Newberry_2020
SNCA_SEQ = (
    "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTK"
    "EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDP"
    "DNEAYEMPSEEGYQDYEPEA"
)
assert len(SNCA_SEQ) == 140, f"Expected L=140, got {len(SNCA_SEQ)}"

MODEL_ID = "facebook/esm2_t33_650M_UR50D"

# 20 standard AAs
AA = "ACDEFGHIKLMNPQRSTVWY"


def _all_single_mutants(sequence: str, n: int) -> list[str]:
    """Generate n single-residue mutants (row by row across positions)."""
    variants = []
    for pos_1, wt in enumerate(sequence, start=1):
        for mt in AA:
            if mt != wt:
                variants.append(f"{wt}{pos_1}{mt}")
            if len(variants) >= n:
                return variants
    return variants


# ── fair-esm scoring ─────────────────────────────────────────────────────────

def score_fair_esm(sequence: str, variants: list[str], device: str) -> list[float]:
    """
    Score using Meta's original fair-esm library.
    Replicates the core of compute_fitness.py --scoring-strategy masked-marginals.

    Source:
      refs/ProteinGym/proteingym/baselines/esm/compute_fitness.py:486-504
      refs/fair-esm/esm/pretrained.py:374
    """
    try:
        import esm as fair_esm
    except ImportError:
        print("fair-esm not installed. Run: pip install fair-esm")
        print("Or: pip install git+https://github.com/facebookresearch/esm.git")
        raise

    print("  Loading ESM-2 650M via fair-esm...")
    model, alphabet = fair_esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    L = batch_tokens.size(1)  # includes BOS + EOS

    print(f"  Scoring {L} positions (masked-marginals, fp32, fair-esm)...")
    cache: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for i in range(1, L - 1):  # skip BOS (0) and EOS (L-1)
            masked = batch_tokens.clone()
            masked[0, i] = alphabet.mask_idx
            logits = model(masked)["logits"]
            cache[i] = torch.log_softmax(logits[0, i].float(), dim=-1).cpu()

    scores = []
    for var in variants:
        wt_aa, pos_1, mt_aa = var[0], int(var[1:-1]), var[-1]
        wt_id = alphabet.get_idx(wt_aa)
        mt_id = alphabet.get_idx(mt_aa)
        lp = cache[pos_1]
        scores.append((lp[mt_id] - lp[wt_id]).item())
    return scores


# ── Our transformers scoring ──────────────────────────────────────────────────

def score_transformers(sequence: str, variants: list[str], device: str) -> list[float]:
    """
    Score using our transformers implementation.
    Replicates scoring/masked_marginals.py but kept inline for direct comparison.

    Source: scoring/masked_marginals.py
    Model:  https://huggingface.co/facebook/esm2_t33_650M_UR50D
    """
    from transformers import AutoTokenizer, EsmForMaskedLM

    print("  Loading ESM-2 650M via transformers (fp16)...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (
        EsmForMaskedLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        .eval()
        .to(device)
    )
    mask_id = tok.mask_token_id

    enc = tok(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    ids_base = enc["input_ids"]

    positions_needed = {int(v[1:-1]) for v in variants}
    print(f"  Scoring {len(positions_needed)} positions (masked-marginals, fp16, transformers)...")
    cache: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for pos_1 in sorted(positions_needed):
            ids = ids_base.clone()
            ids[0, pos_1] = mask_id   # BOS at 0 → AA at pos_1
            logits = model(input_ids=ids).logits[0, pos_1]
            cache[pos_1] = torch.log_softmax(logits.float(), dim=-1).cpu()

    scores = []
    for var in variants:
        wt_aa, pos_1, mt_aa = var[0], int(var[1:-1]), var[-1]
        wt_id = tok.convert_tokens_to_ids(wt_aa)
        mt_id = tok.convert_tokens_to_ids(mt_aa)
        lp = cache[pos_1]
        scores.append((lp[mt_id] - lp[wt_id]).item())
    return scores


# ── fp32 transformers variant ─────────────────────────────────────────────────

def score_transformers_fp32(sequence: str, variants: list[str], device: str) -> list[float]:
    """Same as score_transformers but fp32 — isolates fp16 vs fp32 effect."""
    from transformers import AutoTokenizer, EsmForMaskedLM

    print("  Loading ESM-2 650M via transformers (fp32)...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (
        EsmForMaskedLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        .eval()
        .to(device)
    )
    mask_id = tok.mask_token_id

    enc = tok(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    ids_base = enc["input_ids"]

    positions_needed = {int(v[1:-1]) for v in variants}
    cache: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for pos_1 in sorted(positions_needed):
            ids = ids_base.clone()
            ids[0, pos_1] = mask_id
            logits = model(input_ids=ids).logits[0, pos_1]
            cache[pos_1] = torch.log_softmax(logits.float(), dim=-1).cpu()

    scores = []
    for var in variants:
        wt_aa, pos_1, mt_aa = var[0], int(var[1:-1]), var[-1]
        wt_id = tok.convert_tokens_to_ids(wt_aa)
        mt_id = tok.convert_tokens_to_ids(mt_aa)
        lp = cache[pos_1]
        scores.append((lp[mt_id] - lp[wt_id]).item())
    return scores


# ── Report ────────────────────────────────────────────────────────────────────

def _report(label_a: str, scores_a, label_b: str, scores_b, variants):
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    rho = spearmanr(a, b).correlation
    diff = np.abs(a - b)
    print(f"\n  {label_a}  vs  {label_b}")
    print(f"  Spearman ρ            : {rho:+.6f}  (target ≥ 0.999)")
    print(f"  Max |diff|            : {diff.max():.6f}")
    print(f"  Mean |diff|           : {diff.mean():.6f}")
    print(f"  Median |diff|         : {np.median(diff):.6f}")
    print(f"  % variants within 1e-3: {100*(diff < 1e-3).mean():.1f}%")
    if rho >= 0.999:
        print("  PASS — implementations are consistent")
    else:
        print("  FAIL — investigate outliers below")
        idx_top = np.argsort(diff)[::-1][:10]
        print("\n  Top-10 outliers:")
        print(f"  {'Variant':<12} {label_a:>12} {label_b:>12} {'|diff|':>10}")
        for i in idx_top:
            print(f"  {variants[i]:<12} {a[i]:>12.5f} {b[i]:>12.5f} {diff[i]:>10.5f}")
    return rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200,
                    help="Number of variants to test (max 2660 for SNCA)")
    ap.add_argument("--device", default="cpu",
                    help="Device: cpu | cuda | mps")
    ap.add_argument("--skip-fair-esm", action="store_true",
                    help="Skip fair-esm (if not installed) — only compare fp16 vs fp32")
    args = ap.parse_args()

    print(f"\nConsistency check — ESM-2 650M masked-marginals")
    print(f"Sequence : SNCA α-synuclein  L={len(SNCA_SEQ)}")
    print(f"Variants : {args.n}  Device: {args.device}")
    print(f"\nSources compared:")
    print(f"  [A] fair-esm fp32")
    print(f"      refs/fair-esm/  →  https://github.com/facebookresearch/esm")
    print(f"      Original scoring: refs/ProteinGym/proteingym/baselines/esm/compute_fitness.py:486")
    print(f"  [B] transformers fp16  (our implementation)")
    print(f"      scoring/masked_marginals.py  →  https://huggingface.co/facebook/esm2_t33_650M_UR50D")
    print(f"  [C] transformers fp32  (isolates fp16 effect)")
    print(f"      same as B but float32")
    print()

    variants = _all_single_mutants(SNCA_SEQ, args.n)
    print(f"Generated {len(variants)} variants from SNCA")

    results = {}

    if not args.skip_fair_esm:
        print("\n── [A] fair-esm fp32 ─────────────────────────────────────────")
        results["fair-esm fp32"] = score_fair_esm(SNCA_SEQ, variants, args.device)

    print("\n── [B] transformers fp16 ─────────────────────────────────────")
    results["transformers fp16"] = score_transformers(SNCA_SEQ, variants, args.device)

    print("\n── [C] transformers fp32 ─────────────────────────────────────")
    results["transformers fp32"] = score_transformers_fp32(SNCA_SEQ, variants, args.device)

    print("\n" + "=" * 66)
    print("COMPARISON RESULTS")
    print("=" * 66)

    rhos = {}
    if "fair-esm fp32" in results:
        rhos["A vs B"] = _report(
            "fair-esm fp32", results["fair-esm fp32"],
            "transformers fp16", results["transformers fp16"],
            variants,
        )
        rhos["A vs C"] = _report(
            "fair-esm fp32", results["fair-esm fp32"],
            "transformers fp32", results["transformers fp32"],
            variants,
        )

    rhos["B vs C"] = _report(
        "transformers fp16", results["transformers fp16"],
        "transformers fp32", results["transformers fp32"],
        variants,
    )

    print("\n" + "=" * 66)
    print("INTERPRETATION")
    print("=" * 66)
    if "A vs C" in rhos:
        print(f"  A vs C (fair-esm vs transformers, both fp32):")
        print(f"    ρ={rhos['A vs C']:+.6f}  ← library difference only")
        print(f"    If < 0.999: weights loaded differently or attention path differs")
    if "A vs B" in rhos:
        print(f"  A vs B (fair-esm fp32 vs transformers fp16):")
        print(f"    ρ={rhos['A vs B']:+.6f}  ← library + precision difference")
        print(f"    This is what we need ≥ 0.999 for the ProteinGym run to be valid")
    print(f"  B vs C (transformers fp16 vs fp32):")
    print(f"    ρ={rhos['B vs C']:+.6f}  ← precision difference only")
    print(f"    Expect > 0.9999 for SNCA-length sequences")


if __name__ == "__main__":
    main()
