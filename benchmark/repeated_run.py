"""
Repeated-run harness for timing ESM-2 scoring with proper cold-start management.

Design:
  - N_WARMUP passes fire before any clock starts:
      pass 0: model weights cold (first CUDA kernel launch, graph compilation)
      pass 1: CUDA graphs warmed, caches hot — this is the "real" warm state
  - N_TIMED passes are then clocked individually with:
      wall clock  : time.perf_counter() (host-side, includes H2D/D2H)
      CUDA events : torch.cuda.Event (device-side, excludes Python overhead)
      peak memory : torch.cuda.max_memory_allocated (reset before each run)
  - Returns per-run arrays + mean/std — caller decides what to store

Usage:
    from benchmark.repeated_run import repeated_run
    result = repeated_run(
        score_fn,           # scoring/ref/*.score_variants
        model, alphabet,
        sequence, variants,
        device="cuda",
        n_warmup=2,
        n_timed=5,
    )
    print(result["wall_mean"], result["wall_std"])
"""

import statistics
import time

import torch


def repeated_run(
    score_fn,
    model,
    alphabet,
    sequence: str,
    variants: list[str],
    device: str = "cuda",
    n_warmup: int = 2,
    n_timed: int = 5,
) -> dict:
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    def _run_once():
        return score_fn(model, alphabet, sequence, variants, device)

    # ── Warmup passes ────────────────────────────────────────────────────────
    warmup_wall = []
    for i in range(n_warmup):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run_once()
        if use_cuda:
            torch.cuda.synchronize()
        warmup_wall.append(time.perf_counter() - t0)

    # ── Timed passes ─────────────────────────────────────────────────────────
    wall_times   = []
    cuda_times   = []   # milliseconds, device-side
    peak_mem_mbs = []

    for _ in range(n_timed):
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end   = torch.cuda.Event(enable_timing=True)
            ev_start.record()

        t0 = time.perf_counter()
        scores = _run_once()
        wall_s = time.perf_counter() - t0

        if use_cuda:
            ev_end.record()
            torch.cuda.synchronize()
            cuda_times.append(ev_start.elapsed_time(ev_end))   # ms
            peak_mem_mbs.append(
                torch.cuda.max_memory_allocated(device) / 1024 ** 2
            )

        wall_times.append(wall_s)

    n = len(wall_times)
    return {
        # per-run arrays
        "wall_runs":      wall_times,
        "cuda_ms_runs":   cuda_times   or None,
        "peak_mem_mb_runs": peak_mem_mbs or None,
        "warmup_wall_s":  warmup_wall,
        # aggregates
        "wall_mean":      statistics.mean(wall_times),
        "wall_std":       statistics.stdev(wall_times) if n > 1 else 0.0,
        "cuda_ms_mean":   statistics.mean(cuda_times)   if cuda_times else None,
        "cuda_ms_std":    statistics.stdev(cuda_times)  if len(cuda_times) > 1 else 0.0,
        "peak_mem_mb":    statistics.mean(peak_mem_mbs) if peak_mem_mbs else None,
        "n_warmup":       n_warmup,
        "n_timed":        n_timed,
        # last run scores (for ρ calculation — all runs produce identical scores)
        "scores":         scores,
    }
