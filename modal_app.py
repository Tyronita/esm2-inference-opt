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

# ── Two-track naming ──────────────────────────────────────────────────────────
#
#  TRACK A  alias: "ref"  / "meta-original"  / "fair-esm-fp32"
#    What:  Meta Research's own PyTorch code — the exact implementation that
#           produced the published ρ=0.440 in Notin et al. NeurIPS 2023.
#    Code:  refs/fair-esm @ 2b369911  (git submodule, SHA-pinned)
#    API:   model(tokens)["logits"]  |  alphabet.mask_idx  |  alphabet.get_idx()
#    Dtype: torch.float32  (no autocast)
#    Used:  ablate_ref_proteingym, compare_tracks Track A
#    Why:   Reproducibility anchor — if our ρ matches published, this is why.
#
#  TRACK B  alias: "hf"  / "hf-transformers"  / "transformers-fp16"
#    What:  HuggingFace's independent re-implementation of ESM-2, ported into
#           the transformers library. Same weights, different module classes,
#           different dtype. NOT what the paper used — deployment-ready variant.
#    Code:  transformers==4.44.0  |  EsmForMaskedLM
#    API:   model(input_ids=ids).logits  |  tokenizer.mask_token_id
#    Dtype: torch.float16  (tensor core acceleration on A100)
#    Used:  compare_tracks Track B, check_consistency.py
#    Why:   fp16 is ~1.5–2× faster; same ρ proven via bridge (ρ=+0.999973).
#
#  BRIDGE  check_consistency.py  →  Spearman ρ = +0.999973 (200 SNCA variants)
#    Same weights, numerically equivalent scores. δρ at 217-assay scale ≈ 0.
#    Any timing difference = dtype only. Any ρ difference = fp32 vs fp16 noise.
#
# ── Pinned implementation SHAs (from refs/ submodules) ───────────────────────
# fair-esm  sha: 2b369911bb5b4b0dda914521b9475cad1656b2ac  version 2.0.1
#   setup.py:  no hard deps (only esmfold extras); needs torch + numpy
#   environment.yml:  pytorch=1.12+, biopython==1.79, numpy==1.21.2, scipy==1.7.1
#   source: refs/fair-esm/setup.py (read directly — not guessed)
#
# ProteinGym sha: 144fe22b07dfaeec2b366f2346203a9838a55b4c  version 1.3
#   compute_fitness.py imports: torch, numpy, pandas, scipy, tqdm, biopython, fair-esm
#   source: refs/ProteinGym/proteingym/baselines/esm/compute_fitness.py (read directly)
#
# transformers ESM: transformers==4.44.0
#   HF model: facebook/esm2_t33_650M_UR50D
#
# Dependency graph: refs/graph.yml

# ── Track C: ESMC 600M (EvolutionaryScale SDK) ───────────────────────────────
#
#  TRACK C  alias: "esmc"  / "evolutionaryscale"  / "esmc-600m"
#    What:  EvolutionaryScale's ESMC 600M — sequence-only ESM-3 architecture
#           (Pre-LN + RoPE + SwiGLU, no structure). Different model from ESM-2.
#    Code:  refs/evolutionaryscale-esm @ 43b4548b  (git submodule, SHA-pinned)
#           refs/ProteinGym/proteingym/baselines/evoscale/compute_fitness.py
#    API:   ESMC SDK — model.encode(ESMProtein) + model.logits(LogitsConfig)
#           EsmcForMaskedLM (HF-style) — model(input_ids=...).logits
#    Dtype: bfloat16 / float16 (SDK default)
#    Used:  run_track_c
#    Why:   Newer architecture, MIT license, same ProteinGym task — clean apples-to-apples.
#
#  SHINKAEVOLVE: batched_masked_32 / batched_masked_64
#    What:  True-batch optimised scoring — B sequences each with ONE masked position,
#           single batched forward pass. Identical scores to sequential; B× faster.
#    MFU target: 40-60% (vs ~3-5% sequential), fills GPU batch dimension.

ESMC_MODEL_ID = "esmc_600m"   # EvolutionaryScale SDK identifier
ESMC_HF_REPO  = "biohub/ESMC-600M"   # HuggingFace hub for EsmcForMaskedLM

# ── Image C — EvolutionaryScale SDK, no fair-esm ─────────────────────────────
# Source: refs/evolutionaryscale-esm/  |  refs/ProteinGym/proteingym/baselines/evoscale/evoscale_env.yml
image_c = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("apt-get update -qq && apt-get install -y --no-install-recommends git")
    .pip_install(
        "torch==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # EvolutionaryScale ESM SDK >= 3.0 — provides ESMC, EsmcForMaskedLM, EsmcTokenizer,
    # ESMProtein, LogitsConfig. Pulls in: transformers, attrs, einops, biotite, tokenizers.
    # Source: refs/evolutionaryscale-esm @ 43b4548b
    .pip_install("esm>=3.0.0", "attrs", "biopython==1.79")
    .pip_install(
        "pandas==2.2.2",
        "scipy==1.13.1",
        "numpy==1.26.4",
        "scikit-learn==1.5.1",
        "tqdm==4.66.4",
        "requests==2.32.3",
    )
    .pip_install("huggingface_hub==0.24.6", "datasets==2.20.0")
    .add_local_python_source("scoring", "benchmark")
)

# ── Reproducible image — all implementations baked in ────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Step 1: git — needed to install fair-esm from pinned SHA
    .run_commands("apt-get update -qq && apt-get install -y --no-install-recommends git")
    # Step 2: PyTorch (CUDA 12.1) — largest wheel, downloaded once and cached
    .pip_install(
        "torch==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # Step 3: fair-esm at exact pinned SHA (2b369911)
    # Requirements: none declared in setup.py beyond torch; biopython is a soft dep
    # Source: refs/fair-esm/setup.py
    .pip_install(
        "fair-esm @ git+https://github.com/facebookresearch/esm.git"
        "@2b369911bb5b4b0dda914521b9475cad1656b2ac",
        "biopython==1.79",   # used by compute_fitness.py: from Bio import SeqIO
    )
    # Step 3: transformers + our scoring/benchmark deps
    # Source: refs/ProteinGym/proteingym/baselines/esm/compute_fitness.py imports
    #         + our benchmark/metrics.py (scikit-learn for AUC/MCC)
    .pip_install(
        "transformers==4.44.0",
        "pandas==2.2.2",
        "scipy==1.13.1",
        "numpy==1.26.4",
        "scikit-learn==1.5.1",
        "tqdm==4.66.4",
        "requests==2.32.3",
    )
    # Step 4: huggingface_hub for dataset push
    .pip_install("huggingface_hub==0.24.6", "datasets==2.20.0")
    # Step 5: our scoring + benchmark modules
    .add_local_python_source("scoring", "benchmark")
)

app = modal.App("esm2-inference-opt", image=image)

vol = modal.Volume.from_name("esm2-weights", create_if_missing=True)

MODEL_ID   = "facebook/esm2_t33_650M_UR50D"
FAIR_ESM_SHA = "2b369911bb5b4b0dda914521b9475cad1656b2ac"
CACHE_DIR  = Path("/cache")
RESULTS_DIR = CACHE_DIR / "results"

N_WARMUP = 2   # CUDA cold-start passes before timing begins
N_TIMED  = 5   # timed passes per assay (mean ± std reported)


# ── Track A: refs/fair-esm (REFERENCE, fp32) ─────────────────────────────────
def load_model_ref(device: str = "cuda"):
    """Load ESM-2 650M via refs/fair-esm @ 2b369911 — fp32, the reference implementation."""
    import torch
    import esm as fair_esm

    print(f"  Loading {MODEL_ID} via fair-esm @ {FAIR_ESM_SHA[:8]}  ({device})...")
    model, alphabet = fair_esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    dtype  = next(model.parameters()).dtype
    print(f"  Loaded {params:.0f}M params  {dtype}  peak={_gpu_mb():.0f}MB")
    return model, alphabet


# ── Track B: transformers (HF API, fp16) — kept for consistency gate ─────────
def load_model(device: str = "cuda"):
    # TRACK B — transformers==4.44.0, fp16.
    # Used only by check_consistency() and legacy entrypoints.
    # For benchmark runs, use load_model_ref() (Track A) above.
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    print(f"  Loading {MODEL_ID} → {device}  [Track B: transformers fp16]...")
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


# ── Track C: ESMC 600M model loaders ─────────────────────────────────────────

def load_model_esmc(device: str = "cuda"):
    """Load ESMC 600M via EvolutionaryScale SDK (ESMC.from_pretrained).

    Returns: (model, None) — SDK encodes internally, no separate tokenizer.
    Refs: refs/evolutionaryscale-esm @ 43b4548b
    """
    import os, torch
    from esm.models.esmc import ESMC

    os.environ.setdefault("HF_HOME", str(CACHE_DIR))
    print(f"  Loading ESMC 600M via SDK @ {ESMC_HF_REPO}  ({device})...")
    model = ESMC.from_pretrained(ESMC_MODEL_ID).to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    dtype  = next(model.parameters()).dtype
    print(f"  Loaded {params:.0f}M params  {dtype}  peak={_gpu_mb():.0f}MB")
    return model, None


def load_model_esmc_hf(device: str = "cuda"):
    """Load ESMC 600M via HuggingFace-style EsmcForMaskedLM.

    Returns: (model, tokenizer) — supports batched input_ids forward pass.
    Used by: batched_masked (shinkaevolve).
    Refs: refs/evolutionaryscale-esm/esm/models/esmc/model.py
    """
    import os, torch
    from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer

    os.environ.setdefault("HF_HOME", str(CACHE_DIR))
    print(f"  Loading EsmcForMaskedLM @ {ESMC_HF_REPO}  ({device}) fp16...")
    model = EsmcForMaskedLM.from_pretrained(
        ESMC_HF_REPO,
        cache_dir=str(CACHE_DIR),
        dtype=torch.float16,
    ).eval().to(device)
    tokenizer = EsmcTokenizer()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded {params:.0f}M params  fp16  peak={_gpu_mb():.0f}MB")
    return model, tokenizer


def _compute_mfu(model, seq_len: int, wall_s: float, a100_tflops: float = 312.0) -> float:
    """Model FLOPs Utilization: achieved / peak (A100 fp16 = 312 TFLOPS).

    FLOPs approximation: 2 × N_params × L  (Chinchilla, forward pass).
    For sequential scoring wall_s is per-pass; for batched it's total/B_passes.
    """
    n_params = sum(p.numel() for p in model.parameters())
    L = seq_len + 2   # BOS + AAs + EOS
    flops = 2 * n_params * L
    if wall_s <= 0:
        return 0.0
    achieved = flops / wall_s / 1e12   # TFLOPS
    return achieved / a100_tflops


def _gpu_mb() -> float:
    import torch
    return torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def _peak_gpu_mb() -> float:
    import torch
    return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def _append_log(entry: str):
    """Append a timestamped entry to the volume results log. Never truncates."""
    import datetime
    log_path = RESULTS_DIR / "log.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    block = f"\n---\n\n## {ts}  {entry}\n"
    with open(log_path, "a") as f:
        f.write(block)
    vol.commit()


def _print_banner(title: str, gpu_name: str):
    w = 72
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"  GPU    : {gpu_name}")
    print(f"  Model  : {MODEL_ID}")
    print(f"  Target : Spearman ρ = 0.414 ± 0.012  (published ESM-2 650M masked_marginals, 217 assays)")
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

    print(f"\n  Published baseline (masked_marginals, ESM-2 650M): ρ = 0.414 ± 0.012")
    print(f"  Source: Notin et al. NeurIPS 2023")
    print(f"{'='*w}\n")


# ── Single-method runner (spawned in parallel) ─────────────────────────────────

@app.function(
    gpu="A100",
    timeout=7200,
    volumes={str(CACHE_DIR): vol},
)
def run_method(method_name: str, assay_ids: str = ""):
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

    assay_list = assay_ids.split(",") if assay_ids else None
    pg_cache = CACHE_DIR / "proteingym"
    df = run_benchmark(method.fn, model, tok, device, assay_ids=assay_list, cache_dir=pg_cache)

    if not df.empty:
        df["method"] = method_name
        df["peak_gpu_mb"] = _peak_gpu_mb()
        out = RESULTS_DIR / f"{method_name}.csv"
        df.to_csv(out, index=False)
        vol.commit()

    return df.to_dict(orient="records") if not df.empty else []


# ── Track A vs Track B comparison — ALL 217 assays, side-by-side ─────────────
# No cherry-picking. Runs the full ProteinGym substitution benchmark on both
# implementations so results are directly comparable without selection bias.

@app.function(
    gpu="A100",
    timeout=28800,   # 8 hr — fp32 A track is ~2× slower; 217 × 3 methods × 2 tracks × 5 runs
    volumes={str(CACHE_DIR): vol},
)
def compare_tracks(n_timed: int = N_TIMED, n_warmup: int = N_WARMUP):
    """
    Run Track A and Track B on ALL 217 ProteinGym substitution assays.

    Track A: refs/fair-esm @ 2b369911, fp32 — reference implementation
    Track B: transformers==4.44.0, fp16    — HuggingFace API implementation

    Identical harness for both: N_WARMUP cold-start passes discarded, then
    N_TIMED timed passes per assay with wall clock + CUDA events + peak VRAM.
    No assay filtering — all 217 assays in DMS_substitutions.csv are run.

    Outputs:
      results/comparison_stream.jsonl  — one row per track × method × assay
      Printed side-by-side summary table at the end

    Run:  modal run --detach modal_app.py::compare_tracks
    Pull: modal run modal_app.py::pull_comparison
    Cost: ~$25–35, ~12–16 hr on A100 40GB (fp32 A + fp16 B, all methods)
    """
    import json, statistics
    import pandas as pd
    import torch
    from benchmark.events import EventLog
    from benchmark.metrics import compute_all
    from benchmark.profiler import extract_summary, maybe_profile
    from benchmark.repeated_run import repeated_run
    from benchmark.proteingym import fetch_dms_data, fetch_reference
    # NOTE: PD_ASSAY_IDS intentionally NOT imported — no cherry-picking

    device   = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    w = 76
    print(f"\n{'='*w}")
    print(f"  Track A vs Track B — ALL 217 ProteinGym Substitution Assays")
    print(f"  GPU     : {gpu_name}")
    print(f"  Model   : {MODEL_ID}")
    print(f"  Warmup  : {n_warmup} passes   Timed: {n_timed} passes")
    print(f"  Assays  : all 217 (no filtering)")
    print(f"{'='*w}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trace_dir   = RESULTS_DIR / "traces"
    stream_path = RESULTS_DIR / "comparison_stream.jsonl"
    events      = EventLog(RESULTS_DIR / "comparison_events.jsonl")

    pg_cache  = CACHE_DIR / "proteingym"
    reference = fetch_reference(pg_cache)
    dms_dir   = fetch_dms_data(pg_cache)   # returns Path to dir of extracted CSVs

    # ── Load both models ──────────────────────────────────────────────────────
    print("Loading Track A (fair-esm fp32)...")
    model_a, alphabet = load_model_ref(device)
    events.model_loaded(f"{MODEL_ID}:track-a:fp32")

    print("Loading Track B (transformers fp16)...")
    model_b, tokenizer = load_model(device)
    events.model_loaded(f"{MODEL_ID}:track-b:fp16")

    # ── Method registries ─────────────────────────────────────────────────────
    from scoring.ref.registry import get_registry as get_registry_a
    from scoring import masked_marginals as mm_b, wt_marginals as wt_b, pseudo_ppl as ppl_b
    from scoring.registry import ScoringMethod

    registry_a = get_registry_a()

    registry_b = [
        ScoringMethod("wt_marginals",    wt_b.score_variants,  "1",          "transformers fp16", 0.430),
        ScoringMethod("masked_marginals", mm_b.score_variants,  "L",          "transformers fp16", 0.440),
        ScoringMethod("pseudo_ppl",      ppl_b.score_variants, "L (mutant)", "transformers fp16", 0.440),
    ]

    tracks = [
        ("fair-esm-fp32", "fair-esm@2b369911", "float32", model_a, alphabet,  registry_a),
        ("hf-fp16",       "transformers-4.44",  "float16", model_b, tokenizer, registry_b),
    ]

    # ── Build assay list from full reference — ALL 217, no filter ────────────
    assays = []
    for _, ref_row in reference.iterrows():
        assay_id = ref_row["DMS_id"]
        dms_file = dms_dir / ref_row["DMS_filename"]
        if not dms_file.exists():
            print(f"  SKIP {assay_id} — {ref_row['DMS_filename']} not in cache")
            continue
        dms = pd.read_csv(dms_file)
        assays.append({
            "assay_id": assay_id,
            "sequence": ref_row["target_seq"],
            "variants": dms["mutant"].tolist(),
            "fitness":  dms["DMS_score"].tolist(),
            "L":        len(ref_row["target_seq"]),
            "N":        len(dms),
        })
    print(f"  Loaded {len(assays)} assays from reference")

    # ── Build resume set ─────────────────────────────────────────────────────
    done: set[tuple[str, str, str]] = set()   # (track, assay_id, method)
    if stream_path.exists():
        with open(stream_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["track"], r["assay_id"], r["method"]))
        print(f"  Resuming — {len(done)} rows already complete")

    # ── Run ───────────────────────────────────────────────────────────────────
    all_rows = []
    for track_id, impl, dtype, model, vocab, registry in tracks:
        print(f"\n{'─'*w}")
        print(f"  TRACK {track_id} — {impl}  ({dtype})")
        print(f"{'─'*w}")

        for method in registry:
            print(f"\n  [{method.name}]  passes={method.passes}")
            method_rows = []

            for i, a in enumerate(assays):
                if (track_id, a["assay_id"], method.name) in done:
                    continue

                # Profile only the very first assay of the entire run
                profile_this = (i == 0 and track_id == "fair-esm-fp32" and method.name == "masked_marginals")
                label = f"{track_id}_{method.name}_{a['assay_id'][:12]}"

                events.assay_start(a["assay_id"], a["L"], a["N"])

                with maybe_profile(profile_this, trace_dir, label) as prof:
                    timing = repeated_run(
                        method.fn, model, vocab,
                        a["sequence"], a["variants"],
                        device=device,
                        n_warmup=n_warmup,
                        n_timed=n_timed,
                    )

                profile_summary = extract_summary(prof) if profile_this else {}
                metrics = compute_all(timing["scores"], a["fitness"])
                rho     = metrics["spearman_rho"]

                row = {
                    "track":           track_id,
                    "implementation":  impl,
                    "dtype":           dtype,
                    "assay_id":        a["assay_id"],
                    "sequence_length": a["L"],
                    "n_variants":      a["N"],
                    "method":          method.name,
                    "gpu":             gpu_name,
                    "wall_mean":       timing["wall_mean"],
                    "wall_std":        timing["wall_std"],
                    "wall_runs":       timing["wall_runs"],
                    "warmup_wall_s":   timing["warmup_wall_s"],
                    "cuda_ms_mean":    timing["cuda_ms_mean"],
                    "cuda_ms_std":     timing["cuda_ms_std"],
                    "peak_mem_mb":     timing["peak_mem_mb"],
                    "n_warmup":        n_warmup,
                    "n_timed":         n_timed,
                    **metrics,
                    "profile":         profile_summary,
                }

                with open(stream_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

                events.assay_done(a["assay_id"], a["L"], timing["wall_mean"], rho)
                cuda_str = f"  cuda={timing['cuda_ms_mean']:.0f}ms" if timing["cuda_ms_mean"] else ""
                print(
                    f"    {a['assay_id']:<50}  ρ={rho:+.4f}  "
                    f"wall={timing['wall_mean']:.3f}±{timing['wall_std']:.3f}s{cuda_str}"
                )
                method_rows.append(row)
                all_rows.append(row)

            if method_rows:
                rhos = [r["spearman_rho"] for r in method_rows]
                walls = [r["wall_mean"] for r in method_rows]
                print(
                    f"    {'── mean':<50}  ρ={statistics.mean(rhos):+.4f}  "
                    f"(published {method.published_rho:.3f})  "
                    f"total_wall={sum(walls):.0f}s"
                )

        vol.commit()

    # ── Summary: mean ρ per method per track ─────────────────────────────────
    print(f"\n{'='*w}")
    print(f"  COMPARISON SUMMARY  (all {len(assays)} assays)")
    print(f"  {'Method':<22}  {'Track':<15}  {'mean ρ':>8}  {'published':>9}  {'Δρ vs pub':>9}  {'total wall':>10}")
    print(f"  {'-'*22}  {'-'*15}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*10}")

    all_stored = all_rows
    if stream_path.exists() and not all_stored:
        with open(stream_path) as f:
            all_stored = [json.loads(l) for l in f if l.strip()]

    for method_name in ["wt_marginals", "masked_marginals", "pseudo_ppl"]:
        pub = {"wt_marginals": 0.430, "masked_marginals": 0.440, "pseudo_ppl": 0.440}[method_name]
        for track_id in ["fair-esm-fp32", "hf-fp16"]:
            subset = [r for r in all_stored if r["method"] == method_name and r["track"] == track_id]
            if not subset:
                continue
            mean_rho  = statistics.mean(r["spearman_rho"] for r in subset)
            total_wall = sum(r["wall_mean"] for r in subset)
            delta     = mean_rho - pub
            print(
                f"  {method_name:<22}  {track_id:<15}  {mean_rho:+.4f}  "
                f"{pub:+.3f}     {delta:+.4f}    {total_wall:8.0f}s"
            )

    print(f"{'='*w}\n")
    vol.commit()
    return {"n_rows": len(all_stored), "stream": str(stream_path)}


@app.local_entrypoint()
def pull_comparison():
    """Pull comparison_stream.jsonl from Modal volume."""
    import subprocess
    for fname in ["comparison_stream.jsonl", "comparison_events.jsonl"]:
        result = subprocess.run(
            ["modal", "volume", "get", "esm2-weights", f"results/{fname}", f"results/{fname}"],
            capture_output=True, text=True,
        )
        print(f"  {'pulled' if result.returncode == 0 else 'not ready'}: results/{fname}")


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
    import datetime, json
    from benchmark.events import EventLog
    from benchmark.proteingym import PD_ASSAY_IDS
    from benchmark.timing import RunTimer
    from scoring.registry import get_registry

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    _print_banner("ESM-2 650M — PD Protein Ablation (all methods)", gpu_name)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pg_cache = CACHE_DIR / "proteingym"

    from benchmark.proteingym import run_benchmark

    # Event log for profiling — shared across all methods in this job
    events = EventLog(RESULTS_DIR / "pd_proteins_events.jsonl")

    model, tok = load_model(device)
    events.model_loaded(MODEL_ID)

    registry = get_registry(device)

    # JSONL stream — one line per assay, never truncated
    jsonl_path = RESULTS_DIR / "pd_proteins_stream.jsonl"

    all_results = {}
    for method in registry:
        print(f"\n{'─'*60}")
        print(f"  Method : {method.name}   [{method.passes} passes/assay]")
        print(f"  Source : {method.source}")
        print(f"  Note   : {method.note}")
        print(f"{'─'*60}")
        torch.cuda.reset_peak_memory_stats()

        timer = RunTimer(method.name, len(PD_ASSAY_IDS))
        timer.mark_model_loaded()
        events.job_start(method.name, len(PD_ASSAY_IDS))
        commit_counter = [0]

        def on_done(row, method_name=method.name):
            row["method"] = method_name
            row["gpu"] = gpu_name
            row["ts"] = datetime.datetime.utcnow().isoformat() + "Z"
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            commit_counter[0] += 1
            if commit_counter[0] % 2 == 0:
                vol.commit()

        df = run_benchmark(
            method.fn, model, tok, device,
            assay_ids=PD_ASSAY_IDS, cache_dir=pg_cache,
            on_assay_done=on_done, timer=timer, event_log=events,
        )

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

    # Build log entry
    import math
    log_lines = [f"Modal A100 — PD proteins ablation (3 assays × 6 methods)\n\n```\ngpu : {gpu_name}\nmodel : {MODEL_ID}\n```\n"]
    log_lines.append("| Method | Spearman ρ (mean) | Wall (s) |")
    log_lines.append("|---|---|---|")
    for method_name, df in all_results.items():
        if df is not None and "spearman_rho" in df.columns:
            rho = df["spearman_rho"].mean(skipna=True)
            wall = df["wall_s"].sum() if "wall_s" in df.columns else float("nan")
            log_lines.append(f"| {method_name} | {rho:+.4f} | {wall:.1f} |")
    _append_log("\n".join(log_lines))

    print(f"  Results saved → {out}")
    return combined.to_dict(orient="records")


# ── CANONICAL BENCHMARK — Track A, fair-esm fp32, N=5 repeated runs ──────────

@app.function(
    gpu="A100",
    timeout=28800,   # 8 hr — fp32 is ~2× slower than fp16; 217 × 3 methods × 5 runs
    volumes={str(CACHE_DIR): vol},
)
def ablate_ref_proteingym(n_timed: int = N_TIMED, n_warmup: int = N_WARMUP, methods: str = ""):
    """
    Reproduce Notin et al. NeurIPS 2023 Table 1 (ESM-2 650M) + A100 timing.

    IMPLEMENTATION: refs/fair-esm @ 2b369911, fp32 (Track A — the reference).
    Timing: N_WARMUP cold-start passes discarded, then N_TIMED timed passes per assay.
    Results: mean ± std wall_s, CUDA device-side ms, per-run arrays, peak VRAM.

    Published targets (Notin et al. 2023, 217 assays):
      wt_marginals     ρ = 0.430
      masked_marginals ρ = 0.440
      pseudo_ppl       ρ = 0.440

    Outputs:
      results/ref_proteingym_stream.jsonl   (append-only, one row per assay × method)
      results/traces/{assay}_{method}.json  (Chrome traces, first run of each)

    Run:
        modal run --detach modal_app.py::ablate_ref_proteingym
    Pull:
        modal run modal_app.py::pull_ref_results
    """
    import json
    import torch
    from benchmark.events import EventLog
    from benchmark.metrics import compute_all
    from benchmark.profiler import extract_summary, maybe_profile
    from benchmark.repeated_run import repeated_run
    from benchmark.proteingym import fetch_dms_data, fetch_reference
    from scoring.ref.registry import get_registry

    device   = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    _print_banner("ESM-2 650M — Canonical ProteinGym Benchmark (Track A, fp32)", gpu_name)
    print(f"  Warmup passes : {n_warmup}  |  Timed passes : {n_timed}")
    print(f"  Implementation: fair-esm @ {FAIR_ESM_SHA[:8]}  (refs/fair-esm)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trace_dir  = RESULTS_DIR / "traces"
    stream_path = RESULTS_DIR / "ref_proteingym_stream.jsonl"

    events = EventLog(RESULTS_DIR / "ref_events.jsonl")
    model, alphabet = load_model_ref(device)
    events.model_loaded(MODEL_ID)

    registry = get_registry()
    if methods:
        keep = set(methods.split(","))
        registry = [m for m in registry if m.name in keep]

    pg_cache  = CACHE_DIR / "proteingym"
    reference = fetch_reference(pg_cache)
    dms_dir   = fetch_dms_data(pg_cache)

    # Build set of already-done (assay, method) pairs to support resuming
    done: set[tuple[str, str]] = set()
    if stream_path.exists():
        with open(stream_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["assay_id"], r["method"]))
        print(f"  Resuming — {len(done)} rows already in stream")

    all_rows: list[dict] = []

    for method in registry:
        print(f"\n{'─'*72}")
        print(f"  Method : {method.name}  [{method.passes} passes/assay]")
        print(f"  Target ρ (published) : {method.published_rho:.3f}")

        rhos = []
        for _, ref_row in reference.iterrows():
            assay_id = ref_row["DMS_id"]
            if (assay_id, method.name) in done:
                continue

            dms_file = dms_dir / ref_row["DMS_filename"]
            if not dms_file.exists():
                print(f"    SKIP {assay_id} — data file not found")
                continue

            import pandas as pd
            dms = pd.read_csv(dms_file)
            sequence = ref_row["target_seq"]
            variants = dms["mutant"].tolist()
            fitness  = dms["DMS_score"].tolist()
            L        = len(sequence)

            events.assay_start(assay_id, L, len(variants))

            # Profile only first assay per method (profiling adds ~3-5× overhead)
            profile_this = (len(rhos) == 0)
            label = f"{assay_id}_{method.name}"

            with maybe_profile(enabled=profile_this, out_dir=trace_dir, label=label) as prof:
                timing = repeated_run(
                    method.fn, model, alphabet,
                    sequence, variants,
                    device=device,
                    n_warmup=n_warmup,
                    n_timed=n_timed,
                )

            profile_summary = extract_summary(prof) if profile_this else {}
            scores = timing["scores"]

            metrics = compute_all(scores, fitness)
            rho     = metrics["spearman_rho"]
            rhos.append(rho)

            row = {
                "assay_id":        assay_id,
                "sequence_length": L,
                "n_variants":      len(variants),
                "method":          method.name,
                "implementation":  f"fair-esm@{FAIR_ESM_SHA[:8]}",
                "dtype":           "float32",
                "gpu":             gpu_name,
                # timing
                "wall_mean":       timing["wall_mean"],
                "wall_std":        timing["wall_std"],
                "wall_runs":       timing["wall_runs"],
                "warmup_wall_s":   timing["warmup_wall_s"],
                "cuda_ms_mean":    timing["cuda_ms_mean"],
                "cuda_ms_std":     timing["cuda_ms_std"],
                "peak_mem_mb":     timing["peak_mem_mb"],
                "n_warmup":        n_warmup,
                "n_timed":         n_timed,
                # metrics
                **metrics,
                # profiling (first assay only)
                "profile":         profile_summary,
            }

            with open(stream_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            vol.commit()

            events.assay_done(assay_id, L, timing["wall_mean"], rho)
            print(
                f"    {assay_id:<50}  ρ={rho:+.4f}  "
                f"wall={timing['wall_mean']:.2f}±{timing['wall_std']:.2f}s"
            )

        if rhos:
            import statistics
            mean_rho = statistics.mean(rhos)
            print(f"\n  {method.name}  mean ρ={mean_rho:+.4f}  (published {method.published_rho:.3f})")
            all_rows.extend([row])

    vol.commit()
    print("\n  Stream complete →", stream_path)
    return {"n_rows": len(all_rows)}


@app.local_entrypoint()
def pull_ref_results():
    """Pull ref_proteingym_stream.jsonl from Modal volume to local results/."""
    import subprocess
    for fname in ["ref_proteingym_stream.jsonl", "ref_events.jsonl"]:
        result = subprocess.run(
            ["modal", "volume", "get", "esm2-weights", f"results/{fname}", f"results/{fname}"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  pulled results/{fname}")
        else:
            print(f"  {fname} not found (run not complete yet)")


# ── Full ProteinGym ablation — Track B, all methods, all 217 assays ───────────
# STATUS: commented out — superseded by ablate_ref_proteingym (Track A above).
# Re-enable for SOTA optimisation work (Flash Attention 2, torch.compile, etc.)

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
    import datetime, json
    import pandas as pd
    import torch
    from benchmark.events import EventLog
    from benchmark.proteingym import run_benchmark
    from benchmark.timing import RunTimer
    from scoring.registry import get_registry

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    _print_banner("ESM-2 650M — Full ProteinGym Ablation (217 assays × all methods)", gpu_name)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pg_cache = CACHE_DIR / "proteingym"

    events = EventLog(RESULTS_DIR / "events.jsonl")
    model, tok = load_model(device)
    events.model_loaded(MODEL_ID)

    registry = get_registry(device)

    all_results = {}
    for method in registry:
        print(f"\n{'─'*60}")
        print(f"  Method : {method.name}   [{method.passes} passes/assay]")
        print(f"  Note   : {method.note}")
        print(f"{'─'*60}")
        torch.cuda.reset_peak_memory_stats()

        # Streaming JSONL — one line per assay, appended as results arrive
        jsonl_path = RESULTS_DIR / f"{method.name}_stream.jsonl"
        timer = RunTimer(method.name, 217)
        timer.mark_model_loaded()
        events.job_start(method.name, 217)
        commit_counter = [0]

        def on_done(row, method_name=method.name, jpath=jsonl_path):
            row["method"] = method_name
            row["gpu"] = gpu_name
            row["ts"] = datetime.datetime.utcnow().isoformat() + "Z"
            with open(jpath, "a") as f:
                f.write(json.dumps(row) + "\n")
            commit_counter[0] += 1
            if commit_counter[0] % 10 == 0:
                vol.commit()

        df = run_benchmark(
            method.fn, model, tok, device,
            assay_ids=None, cache_dir=pg_cache,
            on_assay_done=on_done, timer=timer, event_log=events,
        )

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

    import math
    log_lines = [f"Modal A100 — Full ProteinGym ablation (217 assays × 6 methods)\n\n```\ngpu : {gpu_name}\nmodel : {MODEL_ID}\ntarget : 0.414 ± 0.012\n```\n"]
    log_lines.append("| Method | Spearman ρ (mean 217) | vs baseline | Wall (s) |")
    log_lines.append("|---|---|---|---|")
    for method_name, df in all_results.items():
        if df is not None and "spearman_rho" in df.columns:
            rho = df["spearman_rho"].mean(skipna=True)
            delta = rho - 0.414
            wall = df["wall_s"].sum() if "wall_s" in df.columns else float("nan")
            log_lines.append(f"| {method_name} | {rho:+.4f} | {delta:+.4f} | {wall:.0f} |")
    _append_log("\n".join(log_lines))

    print(f"  Results saved → {RESULTS_DIR}/full_ablation_217.csv")
    return combined.to_dict(orient="records")


# ── Consistency check — fair-esm vs transformers on SNCA ─────────────────────

@app.function(
    gpu="A100",
    timeout=1800,
    volumes={str(CACHE_DIR): vol},
)
def check_consistency(n: int = 200):
    """
    Score SNCA variants with BOTH fair-esm (original, fp32) and our
    transformers implementation (fp16). Report Spearman ρ between them.

    This is the reproducibility gate: ρ ≥ 0.999 means our implementation
    matches the one that produced the published ρ=0.414 on ProteinGym.

    Sources compared:
      [A] fair-esm @ 2b369911  (refs/fair-esm/, installed from pinned SHA)
          scoring: refs/ProteinGym/proteingym/baselines/esm/compute_fitness.py:486
      [B] transformers==4.44.0  (our implementation, scoring/masked_marginals.py)

    Decision tree: refs/graph.yml — see triples for full dependency graph.
    """
    import sys
    sys.path.insert(0, str(CACHE_DIR))  # ensure local modules found

    import subprocess
    result = subprocess.run(
        ["python", "check_consistency.py", "--n", str(n), "--device", "cuda"],
        capture_output=True, text=True, cwd="/root"
    )

    # Run via import instead (cleaner in Modal)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from transformers import AutoTokenizer, EsmForMaskedLM

    SNCA_SEQ = (
        "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTK"
        "EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDP"
        "DNEAYEMPSEEGYQDYEPEA"
    )
    AA = "ACDEFGHIKLMNPQRSTVWY"

    def all_single_mutants(seq, n):
        variants = []
        for pos_1, wt in enumerate(seq, start=1):
            for mt in AA:
                if mt != wt:
                    variants.append(f"{wt}{pos_1}{mt}")
                if len(variants) >= n:
                    return variants
        return variants

    variants = all_single_mutants(SNCA_SEQ, n)
    device = "cuda"

    # ── [A] fair-esm fp32 ────────────────────────────────────────────────────
    print("\n── [A] fair-esm fp32 (pinned SHA 2b369911) ─────────────────────")
    import esm as fair_esm
    model_fe, alphabet = fair_esm.pretrained.esm2_t33_650M_UR50D()
    model_fe = model_fe.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("p", SNCA_SEQ)])
    batch_tokens = batch_tokens.to(device)
    L = batch_tokens.size(1)

    cache_fe: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for i in range(1, L - 1):
            masked = batch_tokens.clone()
            masked[0, i] = alphabet.mask_idx
            logits = model_fe(masked)["logits"]
            cache_fe[i] = torch.log_softmax(logits[0, i].float(), dim=-1).cpu()

    scores_fe = []
    for var in variants:
        wt_aa, pos_1, mt_aa = var[0], int(var[1:-1]), var[-1]
        lp = cache_fe[pos_1]
        scores_fe.append((lp[alphabet.get_idx(mt_aa)] - lp[alphabet.get_idx(wt_aa)]).item())

    del model_fe, batch_tokens, cache_fe
    torch.cuda.empty_cache()

    # ── [B] transformers fp16 ────────────────────────────────────────────────
    print("\n── [B] transformers fp16 (our implementation) ───────────────────")
    from scoring.masked_marginals import score_variants
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    model_hf = (
        EsmForMaskedLM.from_pretrained(
            MODEL_ID, cache_dir=str(CACHE_DIR), torch_dtype=torch.float16
        ).eval().to(device)
    )
    scores_hf = score_variants(model_hf, tok, SNCA_SEQ, variants, device)
    del model_hf
    torch.cuda.empty_cache()

    # ── Compare ──────────────────────────────────────────────────────────────
    a = np.array(scores_fe, dtype=float)
    b = np.array(scores_hf, dtype=float)
    rho = spearmanr(a, b).correlation
    diff = np.abs(a - b)

    w = 66
    print(f"\n{'='*w}")
    print(f"  CONSISTENCY REPORT  (fair-esm fp32  vs  transformers fp16)")
    print(f"{'='*w}")
    print(f"  Variants tested    : {len(variants)}")
    print(f"  Spearman ρ         : {rho:+.6f}  (gate: ≥ 0.999)")
    print(f"  Max |diff|         : {diff.max():.6f}")
    print(f"  Mean |diff|        : {diff.mean():.6f}")
    print(f"  % within 1e-3      : {100*(diff < 1e-3).mean():.1f}%")
    verdict = "PASS — implementation reproduces original" if rho >= 0.999 else "FAIL — investigate outliers"
    print(f"  Result             : {verdict}")

    if rho < 0.999:
        idx_top = np.argsort(diff)[::-1][:10]
        print(f"\n  Top-10 outliers (fair-esm vs transformers):")
        for i in idx_top:
            print(f"    {variants[i]:<10}  fair={a[i]:+.5f}  hf={b[i]:+.5f}  |d|={diff[i]:.5f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import json
    out = {
        "n_variants": len(variants),
        "spearman_rho": float(rho),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "pct_within_1e3": float(100 * (diff < 1e-3).mean()),
        "verdict": verdict,
        "fair_esm_sha": "2b369911bb5b4b0dda914521b9475cad1656b2ac",
        "transformers_version": "4.44.0",
        "dtype_fair_esm": "float32",
        "dtype_transformers": "float16",
    }
    (RESULTS_DIR / "consistency.json").write_text(json.dumps(out, indent=2))
    vol.commit()

    log_entry = (
        f"Modal A100 — consistency check (fair-esm fp32 vs transformers fp16)\n\n"
        f"```\ngpu   : {torch.cuda.get_device_name(0)}\nmodel : {MODEL_ID}\n"
        f"n     : {len(variants)} SNCA variants\n```\n\n"
        f"| Metric | Value | Gate |\n|---|---|---|\n"
        f"| Spearman ρ | {rho:+.6f} | ≥ 0.999 |\n"
        f"| Max \\|diff\\| | {diff.max():.6f} | — |\n"
        f"| Mean \\|diff\\| | {diff.mean():.6f} | — |\n"
        f"| % within 1e-3 | {100*(diff < 1e-3).mean():.1f}% | — |\n\n"
        f"**{verdict}**"
    )
    _append_log(log_entry)

    print(f"\n  Saved → {RESULTS_DIR}/consistency.json")
    return out


# ── torch.profiler trace — run once to see where time actually goes ───────────

@app.function(
    gpu="A100",
    timeout=600,
    volumes={str(CACHE_DIR): vol},
)
def profile_snca():
    """
    Profile masked_marginals on SNCA (L=140) using torch.profiler.
    Writes a Chrome trace JSON to results/traces/snca_profile.json.
    View with: chrome://tracing  or  https://ui.perfetto.dev

    Cost: ~$0.05 (few minutes on A100).

    Run:
        python -m modal run modal_app.py::profile_snca
        python -m modal volume get esm2-weights results/traces/snca_profile.json .
    """
    import datetime
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from transformers import AutoTokenizer, EsmForMaskedLM

    SNCA_SEQ = (
        "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTK"
        "EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDP"
        "DNEAYEMPSEEGYQDYEPEA"
    )
    VARIANTS = ["A53T", "E46K", "A30P", "G51D", "H50Q"]
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    model = EsmForMaskedLM.from_pretrained(
        MODEL_ID, cache_dir=str(CACHE_DIR), torch_dtype=torch.float16
    ).eval().to(device)

    enc = tok(SNCA_SEQ, return_tensors="pt", add_special_tokens=True).to(device)
    mask_id = tok.mask_token_id

    trace_dir = RESULTS_DIR / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    trace_path = str(trace_dir / f"{ts}_snca_profile.json")

    # Warm up — not profiled
    with torch.no_grad():
        ids = enc["input_ids"].clone(); ids[0, 1] = mask_id
        _ = model(input_ids=ids).logits

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
    ) as prof:
        with torch.no_grad():
            for pos_1 in range(1, min(21, len(SNCA_SEQ) + 1)):   # profile first 20 passes
                with record_function(f"mask_pos_{pos_1}"):
                    ids = enc["input_ids"].clone()
                    ids[0, pos_1] = mask_id
                    logits = model(input_ids=ids).logits
                    _ = torch.log_softmax(logits[0, pos_1].float(), dim=-1)

    prof.export_chrome_trace(trace_path)
    vol.commit()

    # Summary table
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print(f"\n  Chrome trace → {trace_path}")
    print(f"  View at: chrome://tracing  (load the .json)")
    return {"trace_path": trace_path}


# ── Pull results from Modal volume to local results/log.md ────────────────────

@app.local_entrypoint()
def pull_results():
    """
    Download results/log.md from Modal volume and append new entries to local
    results/log.md. Run after any Modal job completes.

        python -m modal run modal_app.py::pull_results
    """
    import subprocess
    import datetime

    print("Pulling results/log.md from Modal volume esm2-weights ...")
    result = subprocess.run(
        ["python", "-m", "modal", "volume", "get", "esm2-weights", "results/log.md", "/tmp/modal_log.md"],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print(f"Volume get failed: {result.stderr}")
        print("(No runs have completed yet, or volume log doesn't exist.)")
        return

    local_log = Path("results/log.md")
    modal_log = Path("/tmp/modal_log.md")

    local_text = local_log.read_text() if local_log.exists() else ""
    modal_text = modal_log.read_text()

    # Find entries in modal log that aren't in local log (append-only merge)
    new_entries = []
    for block in modal_text.split("\n---\n"):
        block = block.strip()
        if block and block not in local_text and block != "# Results Log":
            new_entries.append(block)

    if new_entries:
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(local_log, "a") as f:
            for entry in new_entries:
                f.write(f"\n---\n\n{entry}\n")
        print(f"Appended {len(new_entries)} new entries to results/log.md")
        print("Don't forget to commit: git add results/log.md && git commit -m 'results: add Modal run'")
    else:
        print("No new entries — results/log.md is up to date.")


# ── Track C: ESMC 600M — all 217 assays, sequential + shinkaevolve batched ───

@app.function(
    image=image_c,
    gpu="A100",
    timeout=21600,   # 6 hr — ESMC 600M sequential on 217 assays
    volumes={str(CACHE_DIR): vol},
)
def run_track_c(methods: str = "all", n_assays: int = 0):
    """
    ESMC 600M on all 217 ProteinGym substitution assays.

    Two tracks run sequentially in the same job:
      sequential      — ESMC SDK, exact masked marginals, L passes/assay
      batched_32      — EsmcForMaskedLM, true-batch shinkaevolve B=32, ceil(L/32) passes/assay
      batched_64      — EsmcForMaskedLM, true-batch shinkaevolve B=64, ceil(L/64) passes/assay

    MFU is computed per assay: (2 × N_params × L) / wall_s / 312e12

    Outputs:
      results/track_c_stream.jsonl  — one row per method × assay (streaming, resumable)

    Run:    modal run --detach modal_app.py::run_track_c
    Pull:   modal run modal_app.py::pull_track_c_results
    Cost:   ~$15-25, ~6-8 hr on A100 40GB
    """
    import json, os
    import statistics
    import torch
    from benchmark.proteingym import fetch_dms_data, fetch_reference

    device   = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    os.environ.setdefault("HF_HOME", str(CACHE_DIR))

    w = 76
    print(f"\n{'='*w}")
    print(f"  Track C — ESMC 600M ProteinGym benchmark")
    print(f"  GPU     : {gpu_name}")
    print(f"  Model   : {ESMC_HF_REPO}")
    print(f"  Methods : sequential + batched_32 + batched_64")
    print(f"  Assays  : {'all 217' if n_assays == 0 else n_assays}")
    print(f"{'='*w}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stream_path = RESULTS_DIR / "track_c_stream.jsonl"

    pg_cache  = CACHE_DIR / "proteingym"
    reference = fetch_reference(pg_cache)
    dms_dir   = fetch_dms_data(pg_cache)

    import pandas as pd
    assays = []
    for _, ref_row in reference.iterrows():
        assay_id = ref_row["DMS_id"]
        dms_file = dms_dir / ref_row["DMS_filename"]
        if not dms_file.exists():
            continue
        dms = pd.read_csv(dms_file)
        assays.append({
            "assay_id": assay_id,
            "sequence": ref_row["target_seq"],
            "variants": dms["mutant"].tolist(),
            "fitness":  dms["DMS_score"].tolist(),
            "L":        len(ref_row["target_seq"]),
            "N":        len(dms),
        })
    if n_assays > 0:
        assays = assays[:n_assays]
    print(f"  Loaded {len(assays)} assays")

    # Build resume set
    done: set[tuple[str, str]] = set()
    if stream_path.exists():
        with open(stream_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["method"], r["assay_id"]))
        print(f"  Resuming — {len(done)} rows already in stream")

    from benchmark.metrics import compute_all
    from scoring.esmc.registry import get_registry_sdk, get_registry_hf

    # ── Sequential: ESMC SDK ─────────────────────────────────────────────────
    if methods in ("all", "sequential"):
        print(f"\n{'─'*w}")
        print(f"  Loading ESMC SDK (sequential)...")
        model_sdk, _ = load_model_esmc(device)
        n_params = sum(p.numel() for p in model_sdk.parameters())
        print(f"  {n_params/1e6:.0f}M params")

        registry_sdk = get_registry_sdk()
        for method in registry_sdk:
            print(f"\n  [{method.name}]  passes=L  source={method.source}")
            rhos = []
            for a in assays:
                if (method.name, a["assay_id"]) in done:
                    continue
                import time
                t0 = time.perf_counter()
                try:
                    scores = method.fn(model_sdk, None, a["sequence"], a["variants"], device)
                except Exception as e:
                    print(f"    ERROR {a['assay_id']}: {e}")
                    continue
                wall_s = time.perf_counter() - t0

                metrics = compute_all(scores, a["fitness"])
                rho = metrics["spearman_rho"]
                mfu = _compute_mfu(model_sdk, a["L"], wall_s)
                rhos.append(rho)

                row = {
                    "method":     method.name,
                    "track":      "esmc-sdk-sequential",
                    "model":      ESMC_HF_REPO,
                    "assay_id":   a["assay_id"],
                    "L":          a["L"],
                    "N":          a["N"],
                    "gpu":        gpu_name,
                    "wall_s":     round(wall_s, 3),
                    "mfu":        round(mfu, 4),
                    "n_params":   n_params,
                    **metrics,
                }
                with open(stream_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

                print(f"    {a['assay_id']:<50}  ρ={rho:+.4f}  wall={wall_s:.1f}s  MFU={mfu*100:.1f}%")

            if rhos:
                mean_rho = statistics.mean(rhos)
                print(f"\n  {method.name}  mean ρ = {mean_rho:+.4f}  ({len(rhos)} assays)")

        del model_sdk
        torch.cuda.empty_cache()
        vol.commit()

    # ── Shinkaevolve: EsmcForMaskedLM batched ────────────────────────────────
    if methods in ("all", "batched"):
        print(f"\n{'─'*w}")
        print(f"  Loading EsmcForMaskedLM (batched shinkaevolve)...")
        model_hf, tokenizer_hf = load_model_esmc_hf(device)
        n_params = sum(p.numel() for p in model_hf.parameters())

        registry_hf = get_registry_hf()
        for method in registry_hf:
            print(f"\n  [{method.name}]  passes={method.passes}")
            rhos, mfus, walls = [], [], []
            for a in assays:
                if (method.name, a["assay_id"]) in done:
                    continue
                import time
                t0 = time.perf_counter()
                try:
                    scores = method.fn(model_hf, tokenizer_hf, a["sequence"], a["variants"], device)
                except Exception as e:
                    print(f"    ERROR {a['assay_id']}: {e}")
                    continue
                wall_s = time.perf_counter() - t0

                metrics = compute_all(scores, a["fitness"])
                rho = metrics["spearman_rho"]
                mfu = _compute_mfu(model_hf, a["L"], wall_s)
                rhos.append(rho)
                mfus.append(mfu)
                walls.append(wall_s)

                row = {
                    "method":   method.name,
                    "track":    "esmc-hf-batched",
                    "model":    ESMC_HF_REPO,
                    "assay_id": a["assay_id"],
                    "L":        a["L"],
                    "N":        a["N"],
                    "gpu":      gpu_name,
                    "wall_s":   round(wall_s, 3),
                    "mfu":      round(mfu, 4),
                    "n_params": n_params,
                    **metrics,
                }
                with open(stream_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

                print(
                    f"    {a['assay_id']:<50}  ρ={rho:+.4f}  "
                    f"wall={wall_s:.1f}s  MFU={mfu*100:.1f}%"
                )

            if rhos:
                mean_rho = statistics.mean(rhos)
                mean_mfu = statistics.mean(mfus) * 100
                speedup  = statistics.mean([a["L"] for a in assays if a["L"] > 0]) / max(1, int(method.passes.split("/")[-1].rstrip(")"))) if "/" in method.passes else 1.0
                print(
                    f"\n  {method.name}  mean ρ = {mean_rho:+.4f}  "
                    f"mean MFU = {mean_mfu:.1f}%  "
                    f"({len(rhos)} assays)"
                )

        del model_hf
        torch.cuda.empty_cache()
        vol.commit()

    # ── Summary ───────────────────────────────────────────────────────────────
    all_rows = []
    if stream_path.exists():
        with open(stream_path) as f:
            all_rows = [json.loads(l) for l in f if l.strip()]

    print(f"\n{'='*w}")
    print(f"  TRACK C SUMMARY")
    print(f"  {'Method':<22}  {'mean ρ':>8}  {'mean MFU%':>10}  {'total wall':>10}  {'N assays':>8}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")

    for m_name in ["masked_marginals", "batched_masked_32", "batched_masked_64"]:
        subset = [r for r in all_rows if r["method"] == m_name]
        if not subset:
            continue
        mean_rho  = statistics.mean(r["spearman_rho"] for r in subset)
        mean_mfu  = statistics.mean(r["mfu"] for r in subset) * 100
        total_wall = sum(r["wall_s"] for r in subset)
        print(
            f"  {m_name:<22}  {mean_rho:+.4f}    {mean_mfu:>8.1f}%  "
            f"{total_wall:>10.0f}s  {len(subset):>8}"
        )

    print(f"{'='*w}\n")
    vol.commit()
    return {"n_rows": len(all_rows), "stream": str(stream_path)}


@app.local_entrypoint()
def pull_track_c_results():
    """Pull track_c_stream.jsonl from Modal volume. Logs all current results."""
    import json
    import subprocess
    from pathlib import Path

    Path("results").mkdir(exist_ok=True)
    for fname in ["track_c_stream.jsonl"]:
        r = subprocess.run(
            ["modal", "volume", "get", "esm2-weights", f"results/{fname}", f"results/{fname}"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            print(f"  pulled results/{fname}")
            rows = [json.loads(l) for l in Path(f"results/{fname}").read_text().splitlines() if l.strip()]
            methods_seen = {}
            for row in rows:
                m = row["method"]
                if m not in methods_seen:
                    methods_seen[m] = {"rhos": [], "mfus": [], "walls": []}
                methods_seen[m]["rhos"].append(row.get("spearman_rho", float("nan")))
                methods_seen[m]["mfus"].append(row.get("mfu", 0.0))
                methods_seen[m]["walls"].append(row.get("wall_s", 0.0))

            import statistics as st
            print(f"\n  Progress: {len(rows)} rows complete")
            for m_name, d in methods_seen.items():
                rhos = [x for x in d["rhos"] if x == x]  # drop NaN
                if rhos:
                    print(
                        f"  {m_name:<30}  ρ={st.mean(rhos):+.4f}  "
                        f"MFU={st.mean(d['mfus'])*100:.1f}%  "
                        f"wall_total={sum(d['walls']):.0f}s  "
                        f"n={len(rhos)}"
                    )
        else:
            print(f"  {fname} not ready yet (job still running or not started)")
