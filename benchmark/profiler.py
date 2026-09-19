"""
torch.profiler wrapper for per-assay kernel-level profiling.

Usage — fires on the first timed run of each assay:
    from benchmark.profiler import maybe_profile

    with maybe_profile(enabled=True, out_dir=Path("profiles"), label="SYUA_masked") as prof:
        scores = score_fn(model, alphabet, sequence, variants, device)

    # prof.key_averages() is available after the context exits
    # Chrome trace written to out_dir/label.json

Design choices:
  - Only profiles run index 0 (first timed pass after warmup) — profiling
    overhead is significant (~3-5× slower) and would corrupt timing stats
  - activities: CPU + CUDA (CUDA kernels, memory copies)
  - record_shapes=True: captures tensor shapes for the matmul/bmm ops
  - with_stack=False: reduces trace size
  - Trace exported as Chrome JSON — open in chrome://tracing or Perfetto
"""

import contextlib
from pathlib import Path

import torch


@contextlib.contextmanager
def maybe_profile(enabled: bool, out_dir: Path, label: str):
    if not enabled:
        yield None
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"{label}.json"

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        yield prof

    prof.export_chrome_trace(str(trace_path))


def _cuda_us(e) -> float:
    """Compatibility shim: torch 2.x renamed cuda_time_total on FunctionEventAvg."""
    for attr in ("cuda_time_total", "self_cuda_time_total", "device_time_total"):
        v = getattr(e, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def extract_summary(prof) -> dict:
    """Pull top kernel stats from a completed profiler run."""
    if prof is None:
        return {}
    avgs = prof.key_averages()
    top = sorted(avgs, key=_cuda_us, reverse=True)[:10]
    return {
        "top_kernels": [
            {
                "key":          e.key,
                "cuda_us":      _cuda_us(e),
                "cpu_us":       getattr(e, "cpu_time_total", 0.0),
                "count":        e.count,
                "self_cuda_us": getattr(e, "self_cuda_time_total", 0.0),
            }
            for e in top
        ],
        "total_cuda_us": sum(_cuda_us(e) for e in avgs),
        "total_cpu_us":  sum(getattr(e, "cpu_time_total", 0.0) for e in avgs),
    }
