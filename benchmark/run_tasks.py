"""
Run all ESMC evaluation tasks using the native MLX model.

Tasks
-----
1. Contact prediction   — P@L on 5 CASP15 proteins (attention-map method)
2. Secondary structure  — Q3/Q8 accuracy (linear probe on embeddings)
3. Stability ΔΔG        — Spearman ρ vs FireProtDB T4 lysozyme

Usage
-----
    cd /Users/nialloleary/esm2-inference-opt
    .venv-esmc/bin/python3 benchmark/run_tasks.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np

from esm.models.esmc import EsmcTokenizer
from esm.models.esmc.config import ESMC_300M_HF_REPO
from scoring.esmc.mlx_esmc import EsmcMLX
from scoring.esmc.tasks import contact, secondary_structure, stability


def load_model():
    print("Loading ESMC-300M (MLX) ...")
    t0 = time.perf_counter()
    model = EsmcMLX.from_pretrained("biohub/ESMC-300M")
    tok   = EsmcTokenizer.from_pretrained(ESMC_300M_HF_REPO)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s\n")
    return model, tok


def banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    model, tok = load_model()
    results = {}

    # ── 1. Contact prediction ─────────────────────────────────────
    banner("Task 1 / 3 — Contact Prediction (P@L)")
    contact_results = contact.run(model, tok)
    results["contact"] = [
        {"pdb": r.pdb_id, "chain": r.chain, "L": r.length,
         "P@L": round(r.p_at_l, 3), "P@2L": round(r.p_at_l2, 3)}
        for r in contact_results
    ]
    if contact_results:
        mean_pal = float(np.mean([r.p_at_l for r in contact_results]))
        results["contact_mean_pal"] = round(mean_pal, 3)
        print(f"\nMean P@L: {mean_pal:.3f}")

    # ── 2. Secondary structure ────────────────────────────────────
    banner("Task 2 / 3 — Secondary Structure (linear probe)")
    ss_result = secondary_structure.run(model, tok)
    results["secondary_structure"] = {
        "Q3": round(ss_result.q3, 3),
        "Q8": round(ss_result.q8, 3),
        "n_proteins": ss_result.n_proteins,
        "n_residues": ss_result.n_residues,
    }

    # ── 3. Stability ΔΔG ──────────────────────────────────────────
    banner("Task 3 / 3 — Stability ΔΔG (FireProtDB T4 lysozyme)")
    stab_result = stability.run(model, tok)
    results["stability"] = {
        "dataset": stab_result.dataset,
        "spearman_r": round(stab_result.spearman_r, 3),
        "spearman_p": f"{stab_result.spearman_p:.2e}",
        "n_mutations": stab_result.n_mutations,
    }

    # ── Summary ───────────────────────────────────────────────────
    banner("Summary")
    print(f"Contact P@L (mean)   : {results.get('contact_mean_pal', 'N/A')}")
    print(f"Secondary structure  : Q3={results['secondary_structure']['Q3']}"
          f"  Q8={results['secondary_structure']['Q8']}")
    print(f"Stability Spearman ρ : {results['stability']['spearman_r']}"
          f"  (n={results['stability']['n_mutations']})")

    out = Path(__file__).parent.parent / "results" / "esmc_mlx_tasks.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results → {out}")


if __name__ == "__main__":
    main()
