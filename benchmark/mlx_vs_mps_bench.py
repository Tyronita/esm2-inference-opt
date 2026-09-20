#!/usr/bin/env python3
"""
benchmark/mlx_vs_mps_bench.py

Head-to-head MLX vs PyTorch MPS inference benchmark.

Tasks
-----
  A) 12 ProteinGym WT sequences  — logits only (no attention)
     Matches the masked-marginals scoring hot-path: one forward pass per masked position.

  B) 12 PDB contact-prediction proteins — with attention maps
     Matches the contact-prediction hot-path where O(L²) attention is materialised.

Methodology
-----------
  - N_WARMUP=2 discarded passes before timing starts
  - N_RUNS=5 timed passes per (model × sequence)
  - MLX: mx.eval() used as GPU-sync barrier (lazy eval completes here)
  - MPS: torch.mps.synchronize() used as GPU-sync barrier
  - Wall-clock measured with time.perf_counter() around the sync barrier
  - All raw timings written to results/bench_<date>.json for audit

Usage
-----
    .venv-esmc/bin/python3 benchmark/mlx_vs_mps_bench.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from datetime import datetime

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np
import pandas as pd
import torch
import urllib.request

sys.path.insert(0, str(Path(__file__).parent.parent))

from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
from esm.models.esmc.config import ESMC_300M_HF_REPO
from scoring.esmc.mlx_esmc import EsmcMLX
from scoring.esmc.tasks.contact import (
    _fetch_pdb, _cb_coords, true_contact_map, attn_contact_map, precision_at_l
)

N_WARMUP = 3   # 3 discarded: Metal shader compiles at each new sequence length
N_RUNS   = 5

# ── Sequence sets ─────────────────────────────────────────────────────────────

def proteingym_sequences() -> list[dict]:
    """12 ProteinGym WT sequences spanning L=70..536."""
    df = pd.read_csv(
        Path(__file__).parent.parent /
        "refs/ProteinGym/reference_files/DMS_substitutions.csv"
    )
    df["L"] = df["target_seq"].str.len()
    buckets = [70, 100, 130, 160, 200, 240, 280, 320, 370, 420, 470, 520]
    rows, used = [], set()
    for lo in buckets:
        sub = df[(df.L >= lo) & (df.L < lo + 60) & (~df.DMS_id.isin(used))]
        if len(sub):
            r = sub.sort_values("L").iloc[0]
            rows.append({"id": r.DMS_id, "L": int(r.L), "seq": r.target_seq})
            used.add(r.DMS_id)
    return rows


# 12 PDB proteins for contact prediction — varied lengths and topologies
CONTACT_TARGETS = [
    ("1UBQ", "A", "ubiquitin",       "α/β",  76),
    ("2PTL", "A", "protein-L",       "α/β",  78),
    ("1TEN", "A", "fibronectin-III", "β",     89),
    ("1SHF", "A", "SH2-domain",      "α/β",  97),
    ("1HRC", "A", "cytochrome-C",    "α",    104),
    ("1APS", "A", "HPr-kinase",      "α/β",  88),
    ("2LZM", "A", "T4-lysozyme",     "α",   164),
    ("1QJO", "A", "Im7-colicin",     "α",    87),
    ("3GB1", "A", "protein-G-B1",    "α/β",  56),
    ("1L2Y", "A", "TC5b-peptide",    "α",    20),
    ("8FBK", "A", "CASP15-T1109",    "mixed",218),
    ("8BSN", "A", "CASP15-T1121",    "α",   311),
]


# ── Timing primitives ─────────────────────────────────────────────────────────

def time_mlx_logits(model, ids: mx.array) -> float:
    """Single timed logits forward pass. Returns wall-clock seconds."""
    t0 = time.perf_counter()
    out = model(ids)
    mx.eval(out)   # ← GPU sync: lazy eval completes here
    return time.perf_counter() - t0


def time_mlx_attentions(model, ids: mx.array) -> tuple[float, list]:
    """Single timed forward pass WITH attention maps. Returns (seconds, attentions)."""
    t0 = time.perf_counter()
    _, _, attentions = model.encode(ids, return_attentions=True)
    mx.eval(*[a for a in attentions if a is not None])  # ← GPU sync
    elapsed = time.perf_counter() - t0
    return elapsed, attentions


def time_mlx_attentions_only(model, ids: mx.array) -> float:
    """Timed attention forward pass — frees arrays immediately to avoid heap buildup."""
    t0 = time.perf_counter()
    _, _, attentions = model.encode(ids, return_attentions=True)
    mx.eval(*[a for a in attentions if a is not None])  # ← GPU sync
    elapsed = time.perf_counter() - t0
    del attentions   # free 30×(1,H,L,L) before next run — prevents accumulation
    return elapsed


def time_mps_logits(model, ids_mps: torch.Tensor) -> float:
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids=ids_mps)
    torch.mps.synchronize()   # ← GPU sync
    return time.perf_counter() - t0


def time_mps_attentions(model, ids_mps: torch.Tensor) -> tuple[float, list]:
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=ids_mps, output_attentions=True)
    torch.mps.synchronize()   # ← GPU sync
    return time.perf_counter() - t0, [a.cpu().float().numpy() for a in out.attentions]


def run_repeats(fn, n_warmup: int, n_runs: int) -> list[float]:
    """Run fn() n_warmup + n_runs times, return last n_runs timings."""
    times = []
    for i in range(n_warmup + n_runs):
        t = fn()
        if i >= n_warmup:
            times.append(t)
    return times


def stats(times: list[float]) -> dict:
    a = np.array(times) * 1000  # → ms
    return {
        "mean_ms":   round(float(a.mean()), 1),
        "std_ms":    round(float(a.std()),  1),
        "min_ms":    round(float(a.min()),  1),
        "max_ms":    round(float(a.max()),  1),
        "runs":      list(np.round(a, 1).tolist()),
    }


# ── Attention wrapper for contact.attn_contact_map ───────────────────────────

class _NpWrap:
    """Lets numpy arrays act as mx.array for attn_contact_map's arr[0] indexing."""
    def __init__(self, arr): self._a = arr
    def __getitem__(self, idx): return self._a[idx]


# ── Main benchmark ────────────────────────────────────────────────────────────

def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log: dict = {"run_id": run_id, "n_warmup": N_WARMUP, "n_runs": N_RUNS,
                 "task_A": [], "task_B": []}

    # ── Load models ──────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  MLX vs MPS benchmark  |  {N_WARMUP} warm-up + {N_RUNS} timed runs")
    print("=" * 70)

    print("\nLoading models ...")
    t0 = time.perf_counter()
    mlx_model = EsmcMLX.from_pretrained("biohub/ESMC-300M")
    tok        = EsmcTokenizer.from_pretrained(ESMC_300M_HF_REPO)
    log["mlx_load_s"] = round(time.perf_counter() - t0, 2)
    print(f"  MLX  loaded in {log['mlx_load_s']:.1f}s")

    t0 = time.perf_counter()
    mps_model = EsmcForMaskedLM.from_pretrained(
        ESMC_300M_HF_REPO, device=torch.device("mps"),
        dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    mps_model.eval()
    log["mps_load_s"] = round(time.perf_counter() - t0, 2)
    print(f"  MPS  loaded in {log['mps_load_s']:.1f}s")

    # ── Task A: ProteinGym sequences (logits only) ───────────────────────────
    print("\n" + "─" * 70)
    print("  TASK A — ProteinGym WT sequences  (logits, no attention)")
    print("─" * 70)
    print(f"  {'Assay':42s} {'L':>5}  "
          f"{'MLX ms':>12}  {'MPS ms':>12}  {'Speedup':>8}")

    pg_seqs = proteingym_sequences()
    for entry in pg_seqs:
        seq = entry["seq"]
        L   = entry["L"]
        enc = tok(seq, return_tensors="pt")
        ids_mx  = mx.array(enc["input_ids"].numpy().astype("int32"))
        ids_mps = enc["input_ids"].to("mps")

        mlx_times = run_repeats(lambda: time_mlx_logits(mlx_model, ids_mx),
                                N_WARMUP, N_RUNS)
        mps_times = run_repeats(lambda: time_mps_logits(mps_model, ids_mps),
                                N_WARMUP, N_RUNS)

        ms  = stats(mlx_times)
        ms2 = stats(mps_times)
        speedup = round(ms2["mean_ms"] / ms["mean_ms"], 2)

        label = entry["id"][:42]
        print(f"  {label:42s} {L:>5}  "
              f"{ms['mean_ms']:>7.1f}±{ms['std_ms']:<4.1f}  "
              f"{ms2['mean_ms']:>7.1f}±{ms2['std_ms']:<4.1f}  "
              f"{speedup:>7.2f}x")

        log["task_A"].append({
            "id": entry["id"], "L": L,
            "mlx": ms, "mps": ms2, "speedup": speedup,
        })

    a_mlx = np.mean([r["mlx"]["mean_ms"] for r in log["task_A"]])
    a_mps = np.mean([r["mps"]["mean_ms"] for r in log["task_A"]])
    print(f"\n  {'MEAN':42s} {'':>5}  "
          f"{a_mlx:>7.1f}       {a_mps:>7.1f}       "
          f"{a_mps/a_mlx:>7.2f}x")

    # ── Task B: Contact prediction proteins (with attention maps) ────────────
    print("\n" + "─" * 70)
    print("  TASK B — Contact prediction proteins  (logits + full attention maps)")
    print("─" * 70)
    print(f"  {'Protein':22s} {'Topo':>5} {'L':>5}  "
          f"{'MLX ms':>12}  {'MPS ms':>12}  {'Speedup':>8}  "
          f"{'P@L MLX':>9}  {'P@L MPS':>9}")

    for pdb_id, chain, name, topo, expected_L in CONTACT_TARGETS:
        try:
            pdb_text      = _fetch_pdb(pdb_id)
            seq, coords   = _cb_coords(pdb_text, chain)
        except Exception as e:
            print(f"  {name:22s}  SKIP ({e})")
            continue

        L   = len(seq)
        enc = tok(seq, return_tensors="pt")
        ids_mx  = mx.array(enc["input_ids"].numpy().astype("int32"))
        ids_mps = enc["input_ids"].to("mps")
        true_cm = true_contact_map(coords)

        # timed runs — MLX uses dedicated fn that frees arrays between calls
        mlx_times  = run_repeats(lambda: time_mlx_attentions_only(mlx_model, ids_mx),
                                 N_WARMUP, N_RUNS)
        mps_times  = run_repeats(lambda: time_mps_attentions(mps_model, ids_mps)[0],
                                 N_WARMUP, N_RUNS)

        # P@L from final run (deterministic — no randomness)
        _, mlx_attn = time_mlx_attentions(mlx_model, ids_mx)
        pred_mlx    = attn_contact_map(mlx_attn)
        pal_mlx     = precision_at_l(pred_mlx, true_cm)

        _, mps_attn_np = time_mps_attentions(mps_model, ids_mps)
        pred_mps       = attn_contact_map([_NpWrap(a) for a in mps_attn_np])
        pal_mps        = precision_at_l(pred_mps, true_cm)

        ms  = stats(mlx_times)
        ms2 = stats(mps_times)
        speedup = round(ms2["mean_ms"] / ms["mean_ms"], 2)

        print(f"  {name:22s} {topo:>5} {L:>5}  "
              f"{ms['mean_ms']:>7.1f}±{ms['std_ms']:<4.1f}  "
              f"{ms2['mean_ms']:>7.1f}±{ms2['std_ms']:<4.1f}  "
              f"{speedup:>7.2f}x  "
              f"{pal_mlx:>9.3f}  {pal_mps:>9.3f}")

        log["task_B"].append({
            "pdb": pdb_id, "chain": chain, "name": name,
            "topology": topo, "L": L,
            "mlx": ms, "mps": ms2, "speedup": speedup,
            "pal_mlx": round(pal_mlx, 3), "pal_mps": round(pal_mps, 3),
        })

    b_mlx = np.mean([r["mlx"]["mean_ms"] for r in log["task_B"]])
    b_mps = np.mean([r["mps"]["mean_ms"] for r in log["task_B"]])
    print(f"\n  {'MEAN':22s} {'':>5} {'':>5}  "
          f"{b_mlx:>7.1f}       {b_mps:>7.1f}       "
          f"{b_mps/b_mlx:>7.2f}x")

    # ── Write log ────────────────────────────────────────────────────────────
    out_path = Path(__file__).parent.parent / "results" / f"bench_{run_id}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(log, indent=2))

    print("\n" + "=" * 70)
    print(f"  Task A (logits):          MLX {a_mlx:.0f}ms  MPS {a_mps:.0f}ms  {a_mps/a_mlx:.2f}x")
    print(f"  Task B (with attentions): MLX {b_mlx:.0f}ms  MPS {b_mps:.0f}ms  {b_mps/b_mlx:.2f}x")
    print(f"  Log → results/bench_{run_id}.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
