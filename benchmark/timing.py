"""
Empirical timing model: t(L) = β × L²

β is the per-(AA)² cost of a single masked-marginals run on a given GPU.
Fit β̂ from observed (L, wall_s) pairs as they stream in; use it to
estimate remaining run time.

Mac M3 MPS baseline (measured):
  SNCA  L=140  wall=16.5s   → β = 8.42e-4 s/(AA)²
  LRRK2 L=2527 wall=5148s  → β = 8.07e-4 s/(AA)²
  median β_MPS ≈ 8.25e-4

Expected A100 speedup ~50× → β_A100 ≈ 1.6e-5 s/(AA)²
"""

from __future__ import annotations

import statistics
import time


class RunTimer:
    """Tracks TTFT, per-assay β, rolling ETA. Thread-safe for single-thread use."""

    def __init__(self, method: str, total_assays: int):
        self.method = method
        self.total = total_assays
        self._job_start: float = time.perf_counter()
        self._model_loaded_ts: float | None = None
        self._first_result_ts: float | None = None
        self._betas: list[float] = []       # β = wall_s / L² per completed assay
        self._done: int = 0
        self._remaining_L2: float = 0.0     # sum of L² for not-yet-started assays

    def mark_model_loaded(self):
        self._model_loaded_ts = time.perf_counter()

    def set_remaining_L2(self, remaining_seq_lengths: list[int]):
        self._remaining_L2 = sum(L * L for L in remaining_seq_lengths)

    def record(self, L: int, wall_s: float):
        if self._first_result_ts is None:
            self._first_result_ts = time.perf_counter()
        self._done += 1
        if wall_s > 0 and L > 0:
            self._betas.append(wall_s / (L * L))
        self._remaining_L2 -= L * L

    @property
    def ttft_s(self) -> float | None:
        if self._first_result_ts is None:
            return None
        return self._first_result_ts - self._job_start

    @property
    def model_load_s(self) -> float | None:
        if self._model_loaded_ts is None:
            return None
        return self._model_loaded_ts - self._job_start

    @property
    def beta_hat(self) -> float | None:
        if len(self._betas) < 2:
            return None
        return statistics.median(self._betas)

    @property
    def eta_s(self) -> float | None:
        b = self.beta_hat
        if b is None or self._remaining_L2 <= 0:
            return None
        return b * self._remaining_L2

    def summary_line(self) -> str:
        parts = [f"{self._done}/{self.total}"]
        if self.beta_hat:
            parts.append(f"β={self.beta_hat:.2e} s/(AA)²")
        if self.eta_s is not None:
            eta_min = self.eta_s / 60
            parts.append(f"ETA {eta_min:.1f}min")
        return "  " + "  ".join(parts)

    def final_report(self) -> dict:
        elapsed = time.perf_counter() - self._job_start
        return {
            "method": self.method,
            "total_assays": self.total,
            "completed": self._done,
            "elapsed_s": round(elapsed, 1),
            "model_load_s": round(self.model_load_s, 2) if self.model_load_s else None,
            "ttft_s": round(self.ttft_s, 2) if self.ttft_s else None,
            "beta_hat": self.beta_hat,
            "beta_n": len(self._betas),
        }
