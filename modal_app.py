"""
ESM-2 650M inference optimisation — Modal app.

Hardware: A100 40GB throughout.
Rationale: ESM-2 was trained on A100 80GB clusters (Lin et al. 2023, Science).
A100 40GB is the standard academic inference GPU for this era and sufficient
for ESM-2 650M (1.3GB fp16). Neither the ProteinGym nor OFS papers specify
a GPU for scoring — A100 is the implied hardware.

Entrypoints:
    modal run modal_app.py::score_pd_proteins      # PD proteins only (~5 min)
    modal run modal_app.py::score_proteingym        # all 217 assays (~2hr)
    modal run modal_app.py::compare_methods         # masked_marginal vs OFS vs wt_marginal
"""

import sys
from pathlib import Path

import modal

# ── Pure reproducible image ───────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.0",
        "pandas==2.2.2",
        "scipy==1.13.1",
        "numpy==1.26.4",
        "tqdm==4.66.4",
        "requests==2.32.3",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # Mount local scoring + benchmark code into the container
    .add_local_python_source("scoring", "benchmark")
)

app = modal.App("esm2-inference-opt", image=image)

# Persistent volume for model weights + ProteinGym cache
# Avoids re-downloading 2.4GB weights on every run
vol = modal.Volume.from_name("esm2-weights", create_if_missing=True)

MODEL_ID = "facebook/esm2_t33_650M_UR50D"
CACHE_DIR = Path("/cache")


def load_model(device: str = "cuda"):
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    model = (
        EsmForMaskedLM.from_pretrained(
            MODEL_ID,
            cache_dir=str(CACHE_DIR),
            torch_dtype=torch.float16,
        )
        .eval()
        .to(device)
    )
    return model, tok


# ── PD proteins — fast smoke test ─────────────────────────────────────────────

@app.function(
    gpu="A100",
    timeout=600,
    volumes={str(CACHE_DIR): vol},
)
def score_pd_proteins():
    """Score PD-relevant variants: SNCA (A53T, E46K, A30P), LRRK2 (G2019S), GBA (N370S)."""
    import torch
    from benchmark.proteingym import PD_ASSAY_IDS, run_benchmark
    from scoring import masked_marginal, ofs, wt_marginal

    device = "cuda"
    model, tok = load_model(device)

    print(f"\n{'='*60}")
    print(f"  ESM-2 650M — PD Protein Scoring")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    methods = {
        "masked_marginal (L passes)": masked_marginal.score_variants,
        "OFS 1-pass             ": ofs.score_variants,
        "wt_marginal (1 pass)   ": wt_marginal.score_variants,
    }

    all_results = {}
    for name, fn in methods.items():
        print(f"\n── {name} ──")
        import time
        t0 = time.perf_counter()
        df = run_benchmark(fn, model, tok, device, assay_ids=PD_ASSAY_IDS)
        elapsed = time.perf_counter() - t0
        df["method"] = name.strip()
        df["wall_s"] = elapsed
        all_results[name] = df
        if not df.empty:
            print(f"  Wall time: {elapsed:.1f}s")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  Summary — mean Spearman ρ across PD assays")
    print(f"  {'Method':<35}  {'ρ':>6}  {'time':>8}")
    print(f"  {'-'*35}  {'-'*6}  {'-'*8}")
    for name, df in all_results.items():
        if not df.empty:
            rho = df["spearman_rho"].mean()
            t   = df["wall_s"].iloc[0]
            print(f"  {name:<35}  {rho:>6.3f}  {t:>7.1f}s")
    print(f"\n  Published ESM-2 650M baseline (full 217 assays): ρ = 0.44")
    print(f"{'='*60}\n")

    vol.commit()
    return {k: v.to_dict(orient="records") for k, v in all_results.items()}


# ── Full ProteinGym (217 assays) ───────────────────────────────────────────────

@app.function(
    gpu="A100",
    timeout=7200,
    volumes={str(CACHE_DIR): vol},
)
def score_proteingym(method: str = "masked_marginal"):
    """
    Full 217-assay ProteinGym substitution benchmark.
    method: 'masked_marginal' | 'ofs' | 'wt_marginal'
    """
    import pandas as pd
    import torch
    from benchmark.proteingym import run_benchmark
    from scoring import masked_marginal, ofs, wt_marginal

    fns = {
        "masked_marginal": masked_marginal.score_variants,
        "ofs":             ofs.score_variants,
        "wt_marginal":     wt_marginal.score_variants,
    }
    assert method in fns, f"method must be one of {list(fns.keys())}"

    device = "cuda"
    model, tok = load_model(device)

    print(f"\n{'='*60}")
    print(f"  ESM-2 650M — Full ProteinGym  [{method}]")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Published baseline (masked marginal): ρ = 0.44")
    print(f"{'='*60}\n")

    df = run_benchmark(fns[method], model, tok, device)

    out_path = CACHE_DIR / f"results_{method}.csv"
    df.to_csv(out_path, index=False)
    vol.commit()

    print(f"\n  Results saved to {out_path}")
    print(f"  Mean ρ = {df['spearman_rho'].mean():.4f}  (n={len(df)} assays)")
    return df.to_dict(orient="records")


# ── Head-to-head comparison (same assays, all three methods) ──────────────────

@app.function(
    gpu="A100",
    timeout=3600,
    volumes={str(CACHE_DIR): vol},
)
def compare_methods(n_assays: int = 20):
    """
    Run all three methods on the same random subset of assays.
    Reproduces Table 1 of Kantroo et al. 2024 for ESM-2 650M.

    Published result: OFS matches masked_marginal within ±0.01 ρ.
    """
    import time

    import pandas as pd
    import torch
    from benchmark.proteingym import fetch_reference, run_benchmark
    from scoring import masked_marginal, ofs, wt_marginal

    device = "cuda"
    model, tok = load_model(device)

    ref = fetch_reference(CACHE_DIR / "proteingym")
    # Sample deterministically — same assays every run
    assay_ids = ref["DMS_id"].sample(n=n_assays, random_state=42).tolist()

    print(f"\n{'='*60}")
    print(f"  Method comparison on {n_assays} assays")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")

    rows = []
    for name, fn in [
        ("wt_marginal",     wt_marginal.score_variants),
        ("ofs",             ofs.score_variants),
        ("masked_marginal", masked_marginal.score_variants),
    ]:
        print(f"\n── {name} ──")
        t0 = time.perf_counter()
        df = run_benchmark(fn, model, tok, device, assay_ids=assay_ids)
        elapsed = time.perf_counter() - t0
        mean_rho = df["spearman_rho"].mean() if not df.empty else float("nan")
        rows.append({"method": name, "mean_rho": mean_rho, "wall_s": elapsed})
        print(f"  Mean ρ = {mean_rho:.4f}  wall = {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"  {'Method':<20}  {'mean ρ':>8}  {'time':>8}  {'speedup vs mm':>14}")
    mm_time = next(r["wall_s"] for r in rows if r["method"] == "masked_marginal")
    for r in rows:
        speedup = f"{mm_time / r['wall_s']:.1f}×" if r["wall_s"] > 0 else "—"
        print(f"  {r['method']:<20}  {r['mean_rho']:>8.4f}  {r['wall_s']:>7.1f}s  {speedup:>14}")
    print(f"\n  Kantroo et al. 2024 claim: OFS ≈ masked_marginal, {n_assays}×+ speedup")
    print(f"{'='*60}\n")

    vol.commit()
    return rows
