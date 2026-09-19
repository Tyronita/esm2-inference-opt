"""
ProteinGym DMS substitution benchmark runner.

Downloads the reference file and DMS CSVs from the official ProteinGym GitHub,
scores each assay with a given scoring function, and reports Spearman ρ.

Published baseline (ESM-2 650M masked marginal): ρ = 0.44
Source: Notin et al. NeurIPS 2023
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
from scipy.stats import spearmanr

REFERENCE_URL = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/"
    "main/reference_files/DMS_substitutions.csv"
)
DMS_ZIP_URL = (
    "https://github.com/OATML-Markslab/ProteinGym/releases/download/"
    "v1.1/DMS_ProteinGym_substitutions.zip"
)

# PD-relevant assays (LewyGym focus)
PD_ASSAY_IDS = [
    "SYUA_HUMAN",   # SNCA — α-synuclein, A53T lives here
    "LRKY_HUMAN",   # LRRK2
    "GBA_HUMAN",    # GBA — glucocerebrosidase
    "PRKN_HUMAN",   # Parkin
    "PINK1_HUMAN",  # PINK1
]


def fetch_reference(cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ref_path = cache_dir / "DMS_substitutions.csv"
    if not ref_path.exists():
        print("Downloading ProteinGym reference file...")
        r = requests.get(REFERENCE_URL, timeout=30)
        r.raise_for_status()
        ref_path.write_bytes(r.content)
    return pd.read_csv(ref_path)


def fetch_dms_data(cache_dir: Path) -> dict[str, pd.DataFrame]:
    """Download and cache all DMS assay CSVs. Returns dict assay_id -> DataFrame."""
    zip_path = cache_dir / "DMS_substitutions.zip"
    dms_dir  = cache_dir / "dms"

    if not dms_dir.exists():
        if not zip_path.exists():
            print(f"Downloading ProteinGym DMS data (~500MB)...")
            r = requests.get(DMS_ZIP_URL, stream=True, timeout=120)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Extracting...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dms_dir)

    assays = {}
    for csv_path in sorted(dms_dir.rglob("*.csv")):
        assays[csv_path.stem] = pd.read_csv(csv_path)
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
    Run ProteinGym benchmark for given assay IDs.

    Args:
        score_fn: callable(model, tokenizer, sequence, variants, device) -> list[float]
        assay_ids: list of assay IDs to run; None = all 217
        cache_dir: where to cache downloaded data

    Returns:
        DataFrame with columns [assay_id, n_variants, spearman_rho, p_value]
    """
    ref = fetch_reference(cache_dir)
    all_assays = fetch_dms_data(cache_dir)

    if assay_ids is None:
        assay_ids = ref["DMS_id"].tolist()

    results = []
    for assay_id in assay_ids:
        if assay_id not in all_assays:
            print(f"  [SKIP] {assay_id} — not found in download")
            continue

        row = ref[ref["DMS_id"] == assay_id]
        if row.empty:
            print(f"  [SKIP] {assay_id} — not in reference")
            continue

        sequence = row["target_seq"].iloc[0]
        dms = all_assays[assay_id]

        variants = dms["mutant"].tolist()
        fitness  = dms["DMS_score"].tolist()

        print(f"  Scoring {assay_id}  L={len(sequence)}  N={len(variants)} variants...", end=" ", flush=True)
        try:
            scores = score_fn(model, tokenizer, sequence, variants, device)
            rho, pval = spearmanr(scores, fitness)
            results.append({
                "assay_id": assay_id,
                "n_variants": len(variants),
                "spearman_rho": rho,
                "p_value": pval,
                "sequence_length": len(sequence),
            })
            print(f"ρ = {rho:.3f}")
        except Exception as e:
            print(f"ERROR: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        print(f"\n  Mean Spearman ρ = {df['spearman_rho'].mean():.3f}  "
              f"(published ESM-2 650M baseline: 0.44)")
    return df
