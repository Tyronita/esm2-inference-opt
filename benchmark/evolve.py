#!/usr/bin/env python3
"""
benchmark/evolve.py  —  MLX optimization evolution loop

Runs through opt levels 0-5 on ESMC-300M, benchmarks each one,
checks correctness (MAE vs fp32), logs JSON, generates a speedup graph,
and commits each level to git.

After all levels are done, enters an extended exploration loop:
  - Varying batch sizes for masked-marginals scoring
  - Longer-sequence throughput curves
  - Per-layer timing analysis (which layers are bottlenecks)
  - Memory profiling across levels

Usage:
    cd /Users/nialloleary/esm2-inference-opt
    .venv-esmc/bin/python3 benchmark/evolve.py [--no-git] [--levels 0,1,2,3]

Outputs:
    results/evolve_log.json        — full timing data
    results/evolve_speedup.png     — bar chart of speedups
    results/evolve_throughput.png  — tokens/sec vs sequence length
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from esm.models.esmc import EsmcTokenizer
from esm.models.esmc.config import ESMC_300M_HF_REPO
from scoring.esmc.mlx_esmc import EsmcMLX

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Benchmark sequences — 12 spanning L=76..480
# ---------------------------------------------------------------------------

BENCH_SEQS = {
    76:  "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    100: "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFMRAFCPGLSPRPATSAAAAAKLSQALQKQLQAAGAFQSAQLQAKQT",
    140: ("MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQ"
          "KTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEM"),
    200: ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQ"
          "TQGFDVTPPLELKGDRAALDQLTSAQQSAIAEQALAMQTEGKPVKMSQNQNGSLAGVNVNASSEQTKQGVSALGEIQK"),
    280: ("MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQ"
          "KTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEAGYMSVAGKVSEACGPAMG"
          "SQSKPGSDLSGASPESRREYQPTAPAQLKLLEATQDVKSGDLVAKDAAAGLN"),
    360: ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQ"
          "TQGFDVTPPLELKGDRAALDQLTSAQQSAIAEQALAMQTEGKPVKMSQNQNGSLAGVNVNASSEQTKQGVSALGEIQK"
          "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQ"
          "KTVEGAGSIAAATGFVKKDQLGKNEEGAP"),
    450: ("MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQ"
          "KTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEAGYMSVAGKVSEACGPAMG"
          "SQSKPGSDLSGASPESRREYQPTAPAQLKLLEATQDVKSGDLVAKMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
          "APILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTQGFDVTPPLELKGDRAAL"),
}


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def bench_logits(model_fn, ids_dict: dict, n_warm: int = 3, n_run: int = 5) -> dict:
    """Warm then time logits-only forward pass. Returns {L: mean_ms}."""
    results = {}
    for L, ids in sorted(ids_dict.items()):
        for _ in range(n_warm):
            o = model_fn(ids); mx.eval(o)
        times = []
        for _ in range(n_run):
            t0 = time.perf_counter()
            o = model_fn(ids); mx.eval(o)
            times.append((time.perf_counter() - t0) * 1000)
        results[L] = {
            "mean_ms": round(float(np.mean(times)), 2),
            "std_ms":  round(float(np.std(times)),  2),
            "min_ms":  round(float(np.min(times)),  2),
            "runs":    [round(t, 2) for t in times],
        }
    return results


def throughput_tokens_per_sec(timing: dict) -> dict:
    """tokens/sec = L / mean_s"""
    return {L: round(L / (v["mean_ms"] / 1000), 0) for L, v in timing.items()}


# ---------------------------------------------------------------------------
# Memory snapshot
# ---------------------------------------------------------------------------

def memory_mb() -> dict:
    return {
        "active":  round(mx.get_active_memory() / 1e6, 1),
        "peak":    round(mx.get_peak_memory()   / 1e6, 1),
        "cache":   round(mx.get_cache_memory()  / 1e6, 1),
    }


# ---------------------------------------------------------------------------
# Correctness check (MAE vs fp32 reference)
# ---------------------------------------------------------------------------

def check_mae(model_fn, ids_fp32_ref: mx.array, logits_ref: np.ndarray) -> float:
    logits_opt = np.array(model_fn(ids_fp32_ref).astype(mx.float32))
    mx.eval()
    return float(np.abs(logits_ref - logits_opt).max())


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_commit(message: str, files: list[str], no_git: bool = False) -> str | None:
    if no_git:
        return None
    try:
        subprocess.run(["git", "add"] + files, cwd=ROOT, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=ROOT, check=True, capture_output=True, text=True
        )
        sha = result.stdout.strip().split("]")[0].split("[")[-1].split(" ")[-1]
        return sha
    except subprocess.CalledProcessError as e:
        print(f"  [git] commit failed: {e.stderr.strip()[:200]}")
        return None


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_graphs(log: dict, out_prefix: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("  [graph] matplotlib not available, skipping")
        return

    levels = log["levels"]
    level_names = [lv["name"] for lv in levels]
    L_keys = sorted({int(k) for lv in levels for k in lv["timing"].keys()})

    # ── 1. Speedup bar chart at each sequence length ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(level_names))
    colors = cm.viridis(np.linspace(0.2, 0.9, len(L_keys)))
    width = 0.8 / len(L_keys)

    baseline_ms = {L: levels[0]["timing"].get(str(L), {}).get("mean_ms", 1.0)
                   for L in L_keys}

    for j, (L, color) in enumerate(zip(L_keys, colors)):
        speedups = []
        for lv in levels:
            ms = lv["timing"].get(str(L), {}).get("mean_ms")
            if ms and ms > 0:
                speedups.append(baseline_ms[L] / ms)
            else:
                speedups.append(0.0)
        offset = (j - len(L_keys) / 2 + 0.5) * width
        bars = ax.bar(x + offset, speedups, width, label=f"L={L}", color=color, alpha=0.85)
        for bar, sp in zip(bars, speedups):
            if sp > 0.1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{sp:.2f}×", ha="center", va="bottom", fontsize=7)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Optimization level")
    ax.set_ylabel("Speedup vs baseline (fp32, no compile)")
    ax.set_title("ESMC-300M MLX Optimization Speedup  |  M3 8GB")
    ax.set_xticks(x)
    ax.set_xticklabels(level_names, rotation=20, ha="right", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, max(1.5, ax.get_ylim()[1] * 1.15))
    plt.tight_layout()
    path = str(out_prefix) + "_speedup.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [graph] speedup chart → {path}")

    # ── 2. Throughput (tokens/sec) vs sequence length per level ──────────
    fig, ax = plt.subplots(figsize=(10, 5))
    colors2 = cm.tab10(np.linspace(0, 0.9, len(levels)))
    for lv, color in zip(levels, colors2):
        Ls = sorted([int(k) for k in lv["timing"].keys()])
        tps = [lv["timing"][str(L)]["mean_ms"] for L in Ls if str(L) in lv["timing"]]
        tok_per_s = [L / (ms / 1000) for L, ms in zip(Ls, tps)]
        ax.plot(Ls, tok_per_s, marker="o", label=lv["name"], color=color, linewidth=2)

    ax.set_xlabel("Sequence length L")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("ESMC-300M MLX Throughput vs Sequence Length")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = str(out_prefix) + "_throughput.png"
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"  [graph] throughput chart → {path2}")

    # ── 3. MAE vs speedup trade-off scatter ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ref_L = sorted(L_keys)[2]  # middle length
    for lv, color in zip(levels, colors2):
        sp = (baseline_ms.get(ref_L, 1.0) /
              max(lv["timing"].get(str(ref_L), {}).get("mean_ms", 1.0), 0.001))
        mae = lv.get("mae", 0.0) or 0.0
        ax.scatter(sp, mae, s=120, color=color, zorder=5, label=lv["name"])
        ax.annotate(lv["name"], (sp, mae), textcoords="offset points",
                    xytext=(6, 3), fontsize=8)

    ax.axhline(2.0, color="red", linestyle="--", linewidth=0.8, alpha=0.7,
               label="MAE=2.0 quality threshold")
    ax.set_xlabel(f"Speedup vs baseline (L={ref_L})")
    ax.set_ylabel("Max absolute error (logits) vs fp32")
    ax.set_title("Quality / Speed Trade-off")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path3 = str(out_prefix) + "_tradeoff.png"
    plt.savefig(path3, dpi=150)
    plt.close()
    print(f"  [graph] trade-off chart → {path3}")


# ---------------------------------------------------------------------------
# Per-layer timing (observability)
# ---------------------------------------------------------------------------

def per_layer_timing(model: EsmcMLX, ids: mx.array, n_run: int = 3) -> list[dict]:
    """Time each of the 30 transformer blocks individually."""
    results = []
    x = model.embed_tokens(ids)
    mx.eval(x)

    for i, block in enumerate(model.layers):
        times = []
        for _ in range(n_run):
            t0 = time.perf_counter()
            x_new, _ = block(x)
            mx.eval(x_new)
            times.append((time.perf_counter() - t0) * 1000)
        results.append({
            "layer": i,
            "mean_ms": round(float(np.mean(times)), 3),
            "std_ms":  round(float(np.std(times)), 3),
        })
        x = x_new  # accumulate for realistic activations

    return results


# ---------------------------------------------------------------------------
# Batched scoring benchmark (ProteinGym hot-path)
# ---------------------------------------------------------------------------

def bench_batched_scoring(model: EsmcMLX, tok, seq: str,
                          batch_sizes: list[int] = (1, 4, 8, 16)) -> dict:
    """Compare sequential vs batched masked-marginals scoring."""
    import string

    # Synthetic mutations: every position → alanine
    mutations = [(i, seq[i], "A") for i in range(min(len(seq), 60))
                 if seq[i] != "A"][:32]
    if not mutations:
        return {}

    results = {}

    for B in batch_sizes:
        if B == 1:
            # Sequential: one pos per forward pass
            t0 = time.perf_counter()
            model.score_variants_batched(tok, seq, mutations, batch_size=1)
            results["sequential"] = round((time.perf_counter() - t0) * 1000, 1)
        else:
            t0 = time.perf_counter()
            model.score_variants_batched(tok, seq, mutations, batch_size=B)
            results[f"batch_{B}"] = round((time.perf_counter() - t0) * 1000, 1)

    return results


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

LEVELS = [
    {"id": 0, "name": "baseline",     "desc": "fp32, no compile"},
    {"id": 1, "name": "mx.compile",   "desc": "fp32 + JIT  (+3%)"},
    {"id": 2, "name": "bf16_only",    "desc": "BF16, no compile  (WARNING: slower without JIT — BF16 dispatch overhead > bandwidth savings)"},
    {"id": 3, "name": "bf16+compile", "desc": "BF16 + JIT  [recommended — best quality/speed]  (+22%)"},
    {"id": 4, "name": "fused_ln",     "desc": "BF16 + JIT + pure-MLX fused residual+LN blocks"},
]


def run_level(level_id: int, tok, ids_dict: dict,
              ref_ids: mx.array, ref_logits: np.ndarray,
              no_git: bool, log: dict) -> dict:
    lvl = LEVELS[level_id]
    print(f"\n{'='*65}")
    print(f"  OPT LEVEL {level_id}: {lvl['name'].upper()}  —  {lvl['desc']}")
    print(f"{'='*65}")

    # Load fresh model for each level (clean state)
    print("  Loading model ...", end=" ", flush=True)
    t_load = time.perf_counter()
    model = EsmcMLX.from_pretrained("biohub/ESMC-300M", opt_level=level_id)
    load_s = time.perf_counter() - t_load
    print(f"{load_s:.1f}s")

    mem = memory_mb()
    print(f"  Memory  active={mem['active']}MB  peak={mem['peak']}MB")

    # MAE correctness check
    print("  Correctness check ...", end=" ", flush=True)
    try:
        mae = check_mae(model, ref_ids, ref_logits)
        mae_ok = mae < 3.0  # BF16 allowed up to ~1.5, int4 higher
        print(f"MAE={mae:.4f} {'✓' if mae_ok else '⚠ HIGH'}")
    except Exception as e:
        mae = None
        mae_ok = False
        print(f"FAILED ({e})")

    # Logits benchmark across all sequence lengths
    print("  Benchmarking logits (3w+5r per length) ...")
    timing = bench_logits(model, ids_dict)
    for L, t in sorted(timing.items()):
        print(f"    L={L:<5d}  {t['mean_ms']:>7.1f} ± {t['std_ms']:.1f} ms")

    # Throughput
    tps = throughput_tokens_per_sec(timing)
    print(f"  Throughput (tok/s): { {L: int(v) for L, v in tps.items()} }")

    # Per-layer timing on L=140 for observability (only baseline + best level)
    layer_timing = None
    if level_id in (0, 3):
        print("  Per-layer timing (L=140) ...")
        ids_140 = ids_dict[140] if 140 in ids_dict else list(ids_dict.values())[2]
        try:
            layer_timing = per_layer_timing(model, ids_140, n_run=3)
            attn_mean = np.mean([l["mean_ms"] for l in layer_timing[:15]])
            ffn_mean  = np.mean([l["mean_ms"] for l in layer_timing[15:]])
            print(f"    Block avg: {np.mean([l['mean_ms'] for l in layer_timing]):.2f} ms  "
                  f"(first-15: {attn_mean:.2f}ms  last-15: {ffn_mean:.2f}ms)")
        except Exception as e:
            print(f"    per-layer failed: {e}")

    # Batched scoring benchmark (level 3+)
    batch_results = {}
    if level_id >= 3:
        print("  Batched masked-marginals scoring ...")
        seq_140 = "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEM"
        try:
            batch_results = bench_batched_scoring(model, tok, seq_140,
                                                   batch_sizes=[1, 4, 8, 16])
            print(f"    {batch_results}")
        except Exception as e:
            print(f"    batched scoring failed: {e}")

    result = {
        "id":            level_id,
        "name":          lvl["name"],
        "desc":          lvl["desc"],
        "load_s":        round(load_s, 2),
        "memory":        mem,
        "mae":           round(mae, 4) if mae is not None else None,
        "mae_ok":        mae_ok,
        "timing":        {str(L): t for L, t in timing.items()},
        "throughput":    {str(L): int(v) for L, v in tps.items()},
        "layer_timing":  layer_timing,
        "batch_scoring": batch_results,
        "timestamp":     datetime.now().isoformat(),
    }

    # Free model memory before logging/graphing
    del model
    mx.clear_cache()

    # Git commit this level
    log_path = RESULTS / "evolve_log.json"
    log["levels"].append(result)
    log_path.write_text(json.dumps(log, indent=2))

    generate_graphs(log, RESULTS / "evolve")

    sha = git_commit(
        f"mlx-opt level {level_id}: {lvl['name']} — "
        f"{list(timing.values())[2]['mean_ms']:.0f}ms @ L={list(timing.keys())[2]}",
        files=["scoring/esmc/mlx_esmc.py",
               "benchmark/evolve.py",
               "results/evolve_log.json",
               "results/evolve_speedup.png",
               "results/evolve_throughput.png",
               "results/evolve_tradeoff.png"],
        no_git=no_git,
    )
    if sha:
        print(f"  [git] committed {sha}")

    return result


# ---------------------------------------------------------------------------
# Extended exploration: batch size sweep
# ---------------------------------------------------------------------------

def extended_batch_sweep(tok, ids_dict, no_git: bool, log: dict):
    """After all levels: sweep batch sizes B=1..16 for level-3 model."""
    print(f"\n{'='*65}")
    print("  EXTENDED: Batched Inference Throughput Sweep")
    print(f"{'='*65}")

    model = EsmcMLX.from_pretrained("biohub/ESMC-300M", opt_level=3)
    L_test = 140
    seq = "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEM"
    ids_single = mx.array(tok(seq, return_tensors="pt")["input_ids"].numpy().astype("int32"))

    print(f"  Sequence length L={L_test}")
    print(f"  {'B':>4}  {'ms/batch':>10}  {'ms/seq':>10}  {'seqs/s':>10}  {'speedup':>10}")

    single_ms = None
    batch_results = []
    for B in [1, 2, 4, 8, 12, 16]:
        try:
            # Build batch by repeating the single sequence
            ids_batch = mx.broadcast_to(ids_single, (B, ids_single.shape[1]))
            for _ in range(3): o = model(ids_batch); mx.eval(o)
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                o = model(ids_batch); mx.eval(o)
                times.append((time.perf_counter() - t0) * 1000)
            batch_ms = float(np.mean(times))
            seq_ms   = batch_ms / B
            seqs_s   = 1000 / seq_ms
            if B == 1:
                single_ms = batch_ms
            speedup = single_ms / seq_ms if single_ms else 1.0
            print(f"  {B:>4}  {batch_ms:>10.1f}  {seq_ms:>10.1f}  {seqs_s:>10.1f}  {speedup:>9.2f}x")
            batch_results.append({
                "B": B, "batch_ms": round(batch_ms, 2),
                "seq_ms": round(seq_ms, 2), "seqs_s": round(seqs_s, 1),
                "speedup": round(speedup, 2)
            })
        except Exception as e:
            print(f"  {B:>4}  FAILED ({e})")

    log["batch_sweep"] = batch_results
    log_path = RESULTS / "evolve_log.json"
    log_path.write_text(json.dumps(log, indent=2))

    # Batch throughput chart
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        Bs  = [r["B"] for r in batch_results]
        tps = [r["seqs_s"] for r in batch_results]
        sps = [r["speedup"] for r in batch_results]
        ax1.bar(range(len(Bs)), tps, color="steelblue")
        ax1.set_xticks(range(len(Bs))); ax1.set_xticklabels([f"B={b}" for b in Bs])
        ax1.set_ylabel("Sequences/sec"); ax1.set_title("Batched Throughput (L=140)")
        ax1.grid(True, alpha=0.3, axis="y")
        ax2.bar(range(len(Bs)), sps, color="darkorange")
        ax2.set_xticks(range(len(Bs))); ax2.set_xticklabels([f"B={b}" for b in Bs])
        ax2.set_ylabel("Per-seq speedup vs B=1"); ax2.set_title("Batching Efficiency")
        ax2.axhline(1.0, linestyle="--", color="gray", alpha=0.5)
        ax2.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(str(RESULTS / "evolve_batch_sweep.png"), dpi=150)
        plt.close()
        print(f"  [graph] batch sweep → results/evolve_batch_sweep.png")
    except Exception:
        pass

    git_commit(
        "mlx-opt: extended batch sweep (B=1..16, L=140)",
        files=["results/evolve_log.json",
               "results/evolve_batch_sweep.png"],
        no_git=no_git,
    )


# ---------------------------------------------------------------------------
# Long-sequence throughput curve
# ---------------------------------------------------------------------------

def extended_length_curve(tok, no_git: bool, log: dict):
    """Throughput vs L from 50 to 1000 for opt levels 0 and 3."""
    print(f"\n{'='*65}")
    print("  EXTENDED: Throughput vs Sequence Length (L=50..1000)")
    print(f"{'='*65}")

    AMINO = "ACDEFGHIKLMNPQRSTVWY"
    lengths = [50, 100, 150, 200, 300, 400, 500, 600, 750, 1000]

    results = {}
    for opt in [0, 3]:
        model = EsmcMLX.from_pretrained("biohub/ESMC-300M", opt_level=opt)
        level_data = []
        print(f"\n  Opt level {opt} ({EsmcMLX.OPT_NAMES[opt]}):")
        for L in lengths:
            import random; random.seed(42)
            seq = "".join(random.choices(AMINO, k=L))
            ids = mx.array(tok(seq, return_tensors="pt")["input_ids"].numpy().astype("int32"))
            try:
                for _ in range(2): o = model(ids); mx.eval(o)
                times = []
                for _ in range(3):
                    t0 = time.perf_counter(); o = model(ids); mx.eval(o)
                    times.append((time.perf_counter()-t0)*1000)
                ms = float(np.mean(times))
                tps = L / (ms / 1000)
                print(f"    L={L:<5d}  {ms:>8.1f} ms  {tps:>7.0f} tok/s")
                level_data.append({"L": L, "ms": round(ms, 1), "tok_s": round(tps, 0)})
            except Exception as e:
                print(f"    L={L:<5d}  FAILED ({e})")
        results[f"opt_{opt}"] = level_data

    log["length_curve"] = results
    RESULTS.joinpath("evolve_log.json").write_text(json.dumps(log, indent=2))

    # Length curve chart
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = {"opt_0": "steelblue", "opt_3": "darkorange"}
        labels = {"opt_0": "baseline (fp32)", "opt_3": "bf16+compile"}
        for key, data in results.items():
            Ls  = [d["L"] for d in data]
            tps = [d["tok_s"] for d in data]
            ax.plot(Ls, tps, marker="o", color=colors[key], label=labels[key], linewidth=2)
        ax.set_xlabel("Sequence length L"); ax.set_ylabel("Throughput (tokens/sec)")
        ax.set_title("ESMC-300M MLX Throughput vs Sequence Length  |  M3 8GB")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(RESULTS / "evolve_length_curve.png"), dpi=150)
        plt.close()
        print(f"\n  [graph] length curve → results/evolve_length_curve.png")
    except Exception:
        pass

    git_commit(
        "mlx-opt: throughput vs length curve (L=50..1000, opt=0,3)",
        files=["results/evolve_log.json", "results/evolve_length_curve.png"],
        no_git=no_git,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-git", action="store_true",
                    help="Skip git commits (useful for dry runs)")
    ap.add_argument("--levels", default="0,1,2,3,4",
                    help="Comma-separated opt levels to run (default: 0,1,2,3,4)")
    ap.add_argument("--skip-extended", action="store_true",
                    help="Skip the extended batch/length sweeps")
    args = ap.parse_args()

    levels_to_run = [int(x) for x in args.levels.split(",")]

    print("=" * 65)
    print("  ESMC-300M  MLX Optimization Evolution Loop")
    print(f"  Device: {mx.device_info()['device_name']}")
    print(f"  MLX: {mx.__version__}")
    print(f"  Levels: {levels_to_run}")
    print("=" * 65)

    tok = EsmcTokenizer.from_pretrained(ESMC_300M_HF_REPO)
    ids_dict = {L: mx.array(tok(s, return_tensors="pt")["input_ids"].numpy().astype("int32"))
                for L, s in BENCH_SEQS.items()}

    # Compute fp32 reference logits for MAE checks
    print("\nComputing fp32 reference logits ...")
    ref_model  = EsmcMLX.from_pretrained("biohub/ESMC-300M", opt_level=0)
    ref_ids    = ids_dict[140]
    ref_logits = np.array(ref_model._raw_forward(ref_ids)); mx.eval()
    del ref_model; mx.clear_cache()
    print("  Done")

    log = {
        "run_started": datetime.now().isoformat(),
        "device": mx.device_info()["device_name"],
        "mlx_version": mx.__version__,
        "levels": [],
    }

    # Main optimization levels
    for lvl_id in levels_to_run:
        try:
            run_level(lvl_id, tok, ids_dict, ref_ids, ref_logits,
                      no_git=args.no_git, log=log)
        except Exception as e:
            print(f"\n  [ERROR] Level {lvl_id} failed: {e}")
            import traceback; traceback.print_exc()

    # Extended sweeps
    if not args.skip_extended:
        try:
            extended_batch_sweep(tok, ids_dict, no_git=args.no_git, log=log)
        except Exception as e:
            print(f"\n  [ERROR] Batch sweep failed: {e}")
            import traceback; traceback.print_exc()

        try:
            extended_length_curve(tok, no_git=args.no_git, log=log)
        except Exception as e:
            print(f"\n  [ERROR] Length curve failed: {e}")
            import traceback; traceback.print_exc()

    # Final summary
    levels_done = log["levels"]
    if levels_done:
        L_ref = 140
        print(f"\n{'='*65}")
        print("  FINAL SUMMARY")
        print(f"{'='*65}")
        print(f"  {'Level':<22}  {'L=140 ms':>10}  {'Speedup':>10}  {'MAE':>8}  {'OK':>4}")
        baseline_ms = next(
            (lv["timing"].get(str(L_ref), {}).get("mean_ms") for lv in levels_done
             if lv["id"] == 0), 1.0
        ) or 1.0
        for lv in levels_done:
            ms   = lv["timing"].get(str(L_ref), {}).get("mean_ms")
            mae  = lv.get("mae")
            spd  = f"{baseline_ms / ms:.2f}x" if ms else "—"
            mae_s = f"{mae:.3f}" if mae is not None else "—"
            ok   = "✓" if lv.get("mae_ok", True) else "⚠"
            print(f"  {lv['name']:<22}  {(str(ms)+'ms'):>10}  {spd:>10}  {mae_s:>8}  {ok:>4}")

    print(f"\n  Log → results/evolve_log.json")
    print(f"  Graphs → results/evolve_speedup.png, evolve_throughput.png")
    print(f"  Branch: mlx-optim-loop  (git log --oneline to review)")


if __name__ == "__main__":
    main()
