#!/usr/bin/env python3
"""
benchmark/bench_serial.py  —  Serial execution of all 10 ESMC-300M optimization levels.

Executes ONE level at a time. Every event is flushed immediately so the
output can be tailed live:

    tail -f results/bench_serial.log

Outputs:
    results/bench_serial.log   — streaming human-readable log
    results/bench_serial.json  — structured results for all levels
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS  = ROOT / "results"
LOG_PATH = RESULTS / "bench_serial.log"
JSON_PATH = RESULTS / "bench_serial.json"
RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Streaming logger — every emit() call flushes both log file and stdout
# ---------------------------------------------------------------------------

_log_file = None

def _init_log() -> None:
    global _log_file
    _log_file = open(LOG_PATH, "w", buffering=1)  # line-buffered

def emit(msg: str, *, prefix: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {prefix}{msg}"
    print(line, flush=True)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()

def emit_json(obj: dict) -> None:
    """Write a structured JSON line to log for downstream parsing."""
    line = "JSON:" + json.dumps(obj)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()

# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

BENCH_SEQ = {
    50:  "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLE",
    76:  "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    140: ("MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTG"
          "VTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEM"),
    280: ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAK"
          "WKRQTQGFDVTPPLELKGDRAALDQLTSAQQSAIAEQALAMQTEGKPVKMSQNQNGSLAGVNVNASSEQTKQGV"
          "SALGEIQKLEELDAERQQRRLLPASQVQPQPPQPHLQPPPQPPHPPPPAQPPQPPAP"),
    450: ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAK"
          "WKRQTQGFDVTPPLELKGDRAALDQLTSAQQSAIAEQALAMQTEGKPVKMSQNQNGSLAGVNVNASSEQTKQGV"
          "SALGEIQKLEELDAERQQRRLLPASQVQPQPPQPHLQPPPQPPHPPPPAQPPQPPAPRPPPAPPRPQPPQPPHA"
          "PQAPAQPQAQPQAQPQAQPQAQPQAQPQAQPQAQPQAQPQAQPQAQPQAQPQPFGA"),
}

REF_IDS_SHORT = None   # filled once


def tokenize_seq(seq: str) -> mx.array:
    """Minimal BOS/EOS tokenization (vocab: A=4, C=5, D=6, E=7, F=8, G=9, H=10,
    I=11, L=12, M=13, K=14, N=15, P=17, Q=16, R=18, S=19, T=20, V=21, W=22, Y=23)"""
    AA = {c: i+4 for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    ids = [1] + [AA.get(c, 3) for c in seq] + [2]
    return mx.array([ids], dtype=mx.int32)


def memory_mb() -> dict[str, float]:
    try:
        return {
            "active": round(mx.metal.get_active_memory() / 1e6, 1),
            "peak":   round(mx.metal.get_peak_memory()   / 1e6, 1),
            "cache":  round(mx.metal.get_cache_memory()  / 1e6, 1),
        }
    except Exception:
        return {"active": 0, "peak": 0, "cache": 0}


def bench_level(level: int, ref_logits: np.ndarray,
                ref_ids: mx.array, n_warm: int = 3, n_run: int = 5) -> dict:
    from scoring.esmc.mlx_esmc import EsmcMLX

    emit(f"Loading model opt={level} ...", prefix="  ")
    t_load = time.perf_counter()
    try:
        model = EsmcMLX.from_pretrained("biohub/ESMC-300M", opt_level=level)
    except Exception as e:
        emit(f"LOAD FAILED: {e}", prefix="  ✗ ")
        return {"level": level, "error": str(e)}
    load_ms = (time.perf_counter() - t_load) * 1000
    emit(f"Loaded in {load_ms/1000:.1f}s", prefix="  ")

    mem = memory_mb()
    emit(f"Memory  active={mem['active']}MB  peak={mem['peak']}MB  cache={mem['cache']}MB", prefix="  ")

    # MAE vs fp32 reference
    try:
        logits = np.array(model(ref_ids).astype(mx.float32))
        mae = float(np.max(np.abs(logits - ref_logits)))
        emit(f"MAE vs fp32: {mae:.4f} {'✓' if mae < 5.0 else '⚠ HIGH'}", prefix="  ")
    except Exception as e:
        emit(f"MAE check failed: {e}", prefix="  ✗ ")
        mae = None

    # Timing per sequence length
    timing: dict[int, dict] = {}
    for L, seq in sorted(BENCH_SEQ.items()):
        ids = tokenize_seq(seq)

        # warmup
        for _ in range(n_warm):
            out = model(ids); mx.eval(out)

        # timed runs
        times = []
        for _ in range(n_run):
            t0 = time.perf_counter()
            out = model(ids); mx.eval(out)
            times.append((time.perf_counter() - t0) * 1000)

        ms_mean = float(np.mean(times))
        ms_std  = float(np.std(times))
        tok_s   = int(L / (ms_mean / 1000))
        timing[L] = {"mean_ms": round(ms_mean, 2), "std_ms": round(ms_std, 2),
                     "tok_s": tok_s}
        emit(f"L={L:<4d}  {ms_mean:6.1f} ± {ms_std:.1f} ms   {tok_s:5d} tok/s", prefix="  ")

    ms140 = timing.get(140, {}).get("mean_ms", 0)
    speedup = round(56.5 / ms140, 3) if ms140 > 0 else 0

    result = {
        "level":   level,
        "load_ms": round(load_ms, 1),
        "memory":  mem,
        "mae":     round(mae, 4) if mae is not None else None,
        "timing":  {str(L): v for L, v in timing.items()},
        "ms140":   round(ms140, 2),
        "speedup": speedup,
    }
    emit(f"Speedup vs fp32 baseline @ L=140: {speedup:.3f}x", prefix="  ★ ")
    emit_json({"type": "level_done", "level": level, "ms140": ms140,
               "speedup": speedup, "mae": mae})

    del model
    mx.clear_cache()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_log()

    emit("=" * 65)
    emit("  ESMC-300M MLX  —  Serial Benchmark  (levels 0-10)")
    emit(f"  Device: {mx.device_info()['device_name']}  "
         f"({mx.device_info()['memory_size']//2**30}GB)")
    emit("=" * 65)

    from scoring.esmc.mlx_esmc import EsmcMLX

    # Build fp32 reference logits once
    emit("Building fp32 reference ...", prefix="  ")
    ref_model = EsmcMLX.from_pretrained("biohub/ESMC-300M", opt_level=0)
    ref_ids = tokenize_seq(BENCH_SEQ[140])  # use L=140 as reference
    ref_logits = np.array(ref_model(ref_ids).astype(mx.float32))
    del ref_model
    mx.clear_cache()
    emit("Reference built.", prefix="  ")

    levels = list(range(0, 11))
    all_results: list[dict] = []

    for level in levels:
        emit("")
        emit("─" * 65)
        emit(f"  OPT LEVEL {level}")
        emit("─" * 65)
        result = bench_level(level, ref_logits, ref_ids)
        all_results.append(result)

        # Save checkpoint after every level
        JSON_PATH.write_text(json.dumps({"run_ts": datetime.now().isoformat(),
                                          "levels": all_results}, indent=2))
        emit(f"Checkpoint saved → {JSON_PATH}", prefix="  ")

    # Final summary table
    emit("")
    emit("=" * 65)
    emit("  FINAL SUMMARY")
    emit("=" * 65)
    emit(f"{'Level':<6} {'Name':<30} {'ms140':>8} {'Speedup':>9} {'MAE':>8}")
    emit("-" * 65)

    names = {
        0: "baseline (fp32)", 1: "mx.compile", 2: "bfloat16",
        3: "bf16+compile", 4: "fused_ln",
        5: "qkv+swiglu fusion", 6: "bits=6 FFN",
        7: "mxfp4 FFN", 8: "bits=3 down_proj",
        9: "fast_layer_norm", 10: "kitchen_sink",
    }
    for r in all_results:
        if "error" in r:
            emit(f"  {r['level']:<4} {'ERROR':<30} {r['error'][:30]}")
            continue
        mae_s = f"{r['mae']:.3f}" if r['mae'] is not None else " n/a"
        emit(f"  {r['level']:<4} {names.get(r['level'],'?'):<30} "
             f"{r.get('ms140',0):>8.1f} {r.get('speedup',0):>9.3f}x {mae_s:>8}")

    emit("")
    emit(f"Full log  → {LOG_PATH}")
    emit(f"JSON data → {JSON_PATH}")


if __name__ == "__main__":
    main()
