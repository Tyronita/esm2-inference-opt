"""
ProteinGym DMS substitution benchmark — data loading and evaluation runner.

Published ESM-2 650M baseline (masked_marginals): mean Spearman ρ = 0.414 ± 0.012
Source: Notin et al. NeurIPS 2023
"""

import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
import torch

from benchmark.metrics import METRIC_LABELS, compute_all

REFERENCE_URL = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/"
    "main/reference_files/DMS_substitutions.csv"
)
DMS_ZIP_URL = (
    "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/"
    "DMS_ProteinGym_substitutions.zip"
)

# PD-relevant assay IDs in ProteinGym 217 substitution benchmark.
# Note: LRRK2 and GBA are NOT in the 217-assay set — not published there.
# PRKN (Parkin/PARK2) is the second PD gene present.
# BRCA1 and ZIKV-NS5 added for length diversity: L=1863, L=3423 (tests L² scaling).
PD_ASSAY_IDS = [
    "SYUA_HUMAN_Newberry_2020",       # SNCA α-synuclein  L=140  (PD, A53T/E46K/A30P)
    "PRKN_HUMAN_Clausen_2023",        # Parkin/PARK2       L=465  (PD)
    "BRCA1_HUMAN_Findlay_2018",       # BRCA1              L=1863 (long, windowing test)
    "A0A140D2T1_ZIKV_Sourisseau_2019",# ZIKV NS5           L=3423 (longest in PG, L² cost)
]


def fetch_reference(cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ref_path = cache_dir / "DMS_substitutions.csv"
    if not ref_path.exists():
        print("  Downloading ProteinGym reference CSV...")
        r = requests.get(REFERENCE_URL, timeout=60)
        r.raise_for_status()
        ref_path.write_bytes(r.content)
    return pd.read_csv(ref_path)


def fetch_dms_data(cache_dir: Path) -> dict[str, pd.DataFrame]:
    zip_path = cache_dir / "DMS_substitutions.zip"
    dms_dir  = cache_dir / "dms_files"

    if not dms_dir.exists():
        if not zip_path.exists():
            print("  Downloading ProteinGym DMS data (~500MB)...")
            r = requests.get(DMS_ZIP_URL, stream=True, timeout=300)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        print("  Extracting DMS files...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dms_dir)

    assays = {}
    for csv_path in sorted(dms_dir.rglob("*.csv")):
        assays[csv_path.stem] = pd.read_csv(csv_path)
    print(f"  Loaded {len(assays)} DMS assays from cache")
    return assays


def run_benchmark(
    score_fn,
    model,
    tokenizer,
    device: str,
    assay_ids: list[str] | None = None,
    cache_dir: Path = Path("/tmp/proteingym_cache"),
    on_assay_done=None,
    timer=None,
    event_log=None,
) -> pd.DataFrame:
    """
    Run ProteinGym benchmark.

    on_assay_done(row: dict) — optional callback fired after each assay.
    Receives the full row dict including metrics and timing. Use for streaming
    JSONL writes or volume commits. Called in-process with ~0ms overhead.

    timer — optional RunTimer instance for TTFT / β tracking.
    event_log — optional EventLog instance for start/stop profiling events.

    Returns DataFrame with all metrics per assay.
    """
    from benchmark.timing import RunTimer

    ref      = fetch_reference(cache_dir)
    all_data = fetch_dms_data(cache_dir)

    if assay_ids is None:
        assay_ids = ref["DMS_id"].tolist()

    # Feed remaining L² into timer for ETA
    if timer is not None:
        seq_lengths = []
        for aid in assay_ids:
            rr = ref[ref["DMS_id"] == aid]
            if not rr.empty:
                seq_lengths.append(len(rr["target_seq"].iloc[0]))
        timer.set_remaining_L2(seq_lengths)

    rows = []
    for assay_id in assay_ids:
        ref_row = ref[ref["DMS_id"] == assay_id]
        if ref_row.empty:
            print(f"  [SKIP] {assay_id} — not in reference")
            continue

        dms_df = None
        for key in all_data:
            if assay_id in key or key in assay_id:
                dms_df = all_data[key]
                break
        if dms_df is None:
            dms_df = all_data.get(assay_id)
        if dms_df is None:
            print(f"  [SKIP] {assay_id} — CSV not found")
            continue

        sequence = ref_row["target_seq"].iloc[0]
        variants = dms_df["mutant"].tolist()
        fitness  = dms_df["DMS_score"].tolist()
        fitness_bin = dms_df["DMS_score_bin"].tolist() if "DMS_score_bin" in dms_df else None

        L, N = len(sequence), len(variants)
        print(f"  {assay_id:<45} L={L:<5} N={N:<6}", end=" ", flush=True)

        if event_log is not None:
            event_log.assay_start(assay_id, L, N)

        t0 = time.perf_counter()
        try:
            scores = score_fn(model, tokenizer, sequence, variants, device)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        wall_s = time.perf_counter() - t0

        unique_positions = len({int(m[1:-1]) for v in variants for m in v.split(":")})
        metrics = compute_all(scores, fitness, fitness_bin)

        row = {
            "assay_id":        assay_id,
            "sequence_length": L,
            "n_variants":      N,
            "wall_s":          round(wall_s, 3),
            "variants_per_s":  round(N / wall_s, 1) if wall_s > 0 else 0,
            "unique_positions": unique_positions,
            "beta":            round(wall_s / (L * L), 8) if L > 0 else None,
            **metrics,
        }
        rows.append(row)

        if timer is not None:
            timer.record(L, wall_s)

        rho = metrics.get("spearman_rho", float("nan"))
        eta_str = f"  {timer.summary_line()}" if timer else ""
        print(f"ρ={rho:+.3f}  {wall_s:.1f}s{eta_str}")

        if event_log is not None:
            event_log.assay_done(assay_id, L, wall_s, metrics.get("spearman_rho", float("nan")))

        if on_assay_done is not None:
            on_assay_done(row)

        if L > 500:
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n  ── Mean metrics across {len(df)} assays ──")
        for col, label in METRIC_LABELS.items():
            if col in df.columns:
                val = df[col].mean(skipna=True)
                print(f"    {label:<22}: {val:+.4f}")
        print(f"    {'Published baseline':<22}: ρ = 0.414 ± 0.012  (ESM-2 650M masked_marginals)")
        if timer is not None:
            r = timer.final_report()
            print(f"\n  ── Timing ──")
            print(f"    model_load           : {r['model_load_s']}s")
            print(f"    TTFT                 : {r['ttft_s']}s")
            bh = r['beta_hat']
            bh_str = f"{bh:.3e}" if bh is not None else "n/a (need ≥2 assays)"
            print(f"    β̂ (median)          : {bh_str} s/(AA)²  (n={r['beta_n']})")
            print(f"    total elapsed        : {r['elapsed_s']}s")
    return df
