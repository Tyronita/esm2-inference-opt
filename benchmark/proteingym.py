"""
ProteinGym DMS substitution benchmark — data loading and evaluation runner.

Published ESM-2 650M baseline (masked_marginals): mean Spearman ρ = 0.44
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
    "https://github.com/OATML-Markslab/ProteinGym/releases/download/"
    "v1.1/DMS_ProteinGym_substitutions.zip"
)

# PD-relevant assay IDs (LewyGym focus)
PD_ASSAY_IDS = [
    "SYUA_HUMAN_Newberry_2020",   # SNCA — α-synuclein; A53T, E46K, A30P live here
    "LRKK2_HUMAN_Zeng_2023",      # LRRK2
    "GBA_HUMAN_Petrosino_2021",   # GBA
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
) -> pd.DataFrame:
    """
    Run ProteinGym benchmark.

    Returns DataFrame with all metrics per assay.
    """
    ref      = fetch_reference(cache_dir)
    all_data = fetch_dms_data(cache_dir)

    if assay_ids is None:
        assay_ids = ref["DMS_id"].tolist()

    rows = []
    for assay_id in assay_ids:
        # Match by DMS_id in reference
        ref_row = ref[ref["DMS_id"] == assay_id]
        if ref_row.empty:
            print(f"  [SKIP] {assay_id} — not in reference")
            continue

        # Find matching CSV (stem may differ from DMS_id)
        dms_df = None
        for key in all_data:
            if assay_id in key or key in assay_id:
                dms_df = all_data[key]
                break
        if dms_df is None:
            # Try exact match
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

        t0 = time.perf_counter()
        try:
            scores = score_fn(model, tokenizer, sequence, variants, device)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        wall_s = time.perf_counter() - t0

        # Count forward passes: unique positions needed
        unique_positions = len({int(m[1:-1]) for v in variants for m in v.split(":")})

        metrics = compute_all(scores, fitness, fitness_bin)
        row = {
            "assay_id":       assay_id,
            "sequence_length": L,
            "n_variants":     N,
            "wall_s":         round(wall_s, 3),
            "variants_per_s": round(N / wall_s, 1) if wall_s > 0 else 0,
            "unique_positions": unique_positions,
            **metrics,
        }
        rows.append(row)

        rho = metrics.get("spearman_rho", float("nan"))
        print(f"ρ={rho:+.3f}  {wall_s:.1f}s")

        # Explicitly free GPU cache between assays for long sequences
        if L > 500:
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n  ── Mean metrics across {len(df)} assays ──")
        for col, label in METRIC_LABELS.items():
            if col in df.columns:
                val = df[col].mean(skipna=True)
                print(f"    {label:<22}: {val:+.4f}")
        print(f"    {'Published baseline':<22}: ρ = 0.44  (ESM-2 650M masked_marginals)")
    return df
