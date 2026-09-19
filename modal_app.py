"""
ESM-2 650M inference optimisation — comprehensive ProteinGym ablation.

Hardware: A100 40GB (implied by Lin et al. 2023 training hardware; neither
ProteinGym nor OFS papers specify inference GPU — see LIMITATIONS.md).

Entrypoints:
    modal run modal_app.py::ablate_pd_proteins      # PD proteins, all methods (~15 min)
    modal run modal_app.py::ablate_proteingym        # Full 217 assays, all methods (~3hr)
    modal run modal_app.py::run_method --method masked_marginals  # Single method, all assays
"""

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
        "scikit-learn==1.5.1",
        "tqdm==4.66.4",
        "requests==2.32.3",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_python_source("scoring", "benchmark")
)

app = modal.App("esm2-inference-opt", image=image)

vol = modal.Volume.from_name("esm2-weights", create_if_missing=True)

MODEL_ID = "facebook/esm2_t33_650M_UR50D"
CACHE_DIR = Path("/cache")
RESULTS_DIR = CACHE_DIR / "results"


def load_model(device: str = "cuda"):
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    print(f"  Loading {MODEL_ID} → {device}...")
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
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded {params:.0f}M params  fp16  peak={_gpu_mb():.0f}MB")
    return model, tok


def _gpu_mb() -> float:
    import torch
    return torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def _peak_gpu_mb() -> float:
    import torch
    return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def _print_banner(title: str, gpu_name: str):
    w = 72
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"  GPU    : {gpu_name}")
    print(f"  Model  : {MODEL_ID}")
    print(f"  Target : Spearman ρ ≥ 0.44  (published ESM-2 650M masked_marginals baseline)")
    print(f"  Ref    : Notin et al. NeurIPS 2023 / Meier et al. NeurIPS 2021")
    print(f"{'='*w}\n")


def _print_summary(all_results: dict[str, dict]):
    """Print consolidated comparison table across all methods."""
    import math

    metrics_to_show = [
        ("spearman_rho",    "Spearman ρ",    "{:+.4f}"),
        ("pearson_r",       "Pearson r",     "{:+.4f}"),
        ("kendall_tau",     "Kendall τ",     "{:+.4f}"),
        ("ndcg_10",         "NDCG@10%",      "{:.4f}"),
        ("ndcg_50",         "NDCG@50%",      "{:.4f}"),
        ("top5_recall",     "Top-5% recall", "{:.4f}"),
        ("top10_recall",    "Top-10% recall","{:.4f}"),
        ("fraction_correct","Frac. correct", "{:.4f}"),
        ("auc",             "AUC-ROC",       "{:.4f}"),
    ]

    timing_cols = [
        ("wall_s",         "Wall (s)",       "{:.1f}"),
        ("variants_per_s", "Variants/s",     "{:.1f}"),
    ]

    methods = list(all_results.keys())
    w = 72

    print(f"\n{'='*w}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*w}")

    # Header
    col_w = 14
    hdr = f"  {'Metric':<22}" + "".join(f"{m[:col_w]:>{col_w}}" for m in methods)
    print(hdr)
    print(f"  {'-'*22}" + f"{'-'*col_w}" * len(methods))

    for col, label, fmt in metrics_to_show:
        row = f"  {label:<22}"
        for method in methods:
            df = all_results[method]
            if df is not None and col in df.columns:
                val = df[col].mean(skipna=True)
                cell = fmt.format(val) if not math.isnan(val) else "   n/a"
            else:
                cell = "   n/a"
            row += f"{cell:>{col_w}}"
        print(row)

    print(f"  {'-'*22}" + f"{'-'*col_w}" * len(methods))

    for col, label, fmt in timing_cols:
        row = f"  {label:<22}"
        for method in methods:
            df = all_results[method]
            if df is not None and col in df.columns:
                val = df[col].sum() if col == "wall_s" else df[col].mean(skipna=True)
                cell = fmt.format(val)
            else:
                cell = "   n/a"
            row += f"{cell:>{col_w}}"
        print(row)

    print(f"\n  Passes per assay (avg sequence length):")
    for method in methods:
        df = all_results[method]
        if df is not None and "unique_positions" in df.columns:
            avg_pos = df["unique_positions"].mean()
            avg_L   = df["sequence_length"].mean()
            print(f"    {method:<30}  unique_pos/assay={avg_pos:.0f}  avg L={avg_L:.0f}")

    print(f"\n  Published baseline (masked_marginals, ESM-2 650M): ρ = 0.44")
    print(f"  Source: Notin et al. NeurIPS 2023")
    print(f"{'='*w}\n")


# ── Single-method runner (spawned in parallel) ─────────────────────────────────

@app.function(
    gpu="A100",
    timeout=7200,
    volumes={str(CACHE_DIR): vol},
)
def run_method(method_name: str, assay_ids: list[str] | None = None):
    """Run one scoring method across given assays. Returns serialisable dict."""
    import torch
    from benchmark.proteingym import PD_ASSAY_IDS, fetch_dms_data, fetch_reference, run_benchmark
    from scoring.registry import get_registry

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()

    model, tok = load_model(device)

    registry = {m.name: m for m in get_registry(device)}
    if method_name not in registry:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(registry.keys())}")

    method = registry[method_name]
    print(f"\n── {method.name}  [{method.passes} passes/assay] ──")
    print(f"   Source : {method.source}")
    print(f"   Note   : {method.note}\n")

    pg_cache = CACHE_DIR / "proteingym"
    df = run_benchmark(method.fn, model, tok, device, assay_ids=assay_ids, cache_dir=pg_cache)

    if not df.empty:
        df["method"] = method_name
        df["peak_gpu_mb"] = _peak_gpu_mb()
        out = RESULTS_DIR / f"{method_name}.csv"
        df.to_csv(out, index=False)
        vol.commit()

    return df.to_dict(orient="records") if not df.empty else []


# ── PD proteins ablation — all methods, fast ──────────────────────────────────

@app.function(
    gpu="A100",
    timeout=3600,
    volumes={str(CACHE_DIR): vol},
)
def ablate_pd_proteins():
    """All scoring methods on PD-relevant assays. Fast (~15 min on A100)."""
    import pandas as pd
    import torch
    from benchmark.proteingym import PD_ASSAY_IDS
    from scoring.registry import get_registry

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    _print_banner("ESM-2 650M — PD Protein Ablation (all methods)", gpu_name)

    model, tok = load_model(device)
    registry = get_registry(device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pg_cache = CACHE_DIR / "proteingym"

    from benchmark.proteingym import run_benchmark

    all_results = {}
    for method in registry:
        print(f"\n{'─'*60}")
        print(f"  Method : {method.name}")
        print(f"  Passes : {method.passes}")
        print(f"  Source : {method.source}")
        print(f"  Note   : {method.note}")
        print(f"{'─'*60}")
        torch.cuda.reset_peak_memory_stats()

        df = run_benchmark(method.fn, model, tok, device, assay_ids=PD_ASSAY_IDS, cache_dir=pg_cache)

        if not df.empty:
            df["method"]       = method.name
            df["peak_gpu_mb"]  = _peak_gpu_mb()
            df["passes_label"] = method.passes
        all_results[method.name] = df if not df.empty else None

        torch.cuda.empty_cache()

    # Consolidated CSV
    combined = pd.concat([df for df in all_results.values() if df is not None], ignore_index=True)
    out = RESULTS_DIR / "pd_proteins_ablation.csv"
    combined.to_csv(out, index=False)
    vol.commit()

    _print_summary(all_results)
    print(f"  Results saved → {out}")

    return combined.to_dict(orient="records")


# ── Full ProteinGym ablation — all methods, all 217 assays ────────────────────

@app.function(
    gpu="A100",
    timeout=14400,
    volumes={str(CACHE_DIR): vol},
)
def ablate_proteingym():
    """
    All scoring methods on all 217 ProteinGym substitution assays.
    Reproduces and extends Table 1 of Notin et al. NeurIPS 2023.
    Runtime: ~3-4 hr on A100 40GB (dominated by masked_marginals and pseudo_ppl).
    """
    import pandas as pd
    import torch
    from benchmark.proteingym import run_benchmark
    from scoring.registry import get_registry

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    _print_banner("ESM-2 650M — Full ProteinGym Ablation (217 assays × all methods)", gpu_name)

    model, tok = load_model(device)
    registry = get_registry(device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pg_cache = CACHE_DIR / "proteingym"

    all_results = {}
    for method in registry:
        print(f"\n{'─'*60}")
        print(f"  Method : {method.name}   [{method.passes} passes/assay]")
        print(f"  Note   : {method.note}")
        print(f"{'─'*60}")
        torch.cuda.reset_peak_memory_stats()

        df = run_benchmark(method.fn, model, tok, device, assay_ids=None, cache_dir=pg_cache)

        if not df.empty:
            df["method"]       = method.name
            df["peak_gpu_mb"]  = _peak_gpu_mb()
            df["passes_label"] = method.passes
            df.to_csv(RESULTS_DIR / f"{method.name}_all217.csv", index=False)
            vol.commit()
        all_results[method.name] = df if not df.empty else None

        torch.cuda.empty_cache()

    combined = pd.concat([df for df in all_results.values() if df is not None], ignore_index=True)
    combined.to_csv(RESULTS_DIR / "full_ablation_217.csv", index=False)
    vol.commit()

    _print_summary(all_results)
    print(f"  Results saved → {RESULTS_DIR}/full_ablation_217.csv")

    return combined.to_dict(orient="records")
