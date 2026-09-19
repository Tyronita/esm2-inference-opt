"""
Lightweight event log for profiling Modal runs.
Append-only JSONL — one JSON object per line.
"""

import json
import time
from pathlib import Path
from typing import Literal

EventKind = Literal[
    "job_start", "model_loaded",
    "assay_start", "assay_done",
    "job_end",
]


class EventLog:
    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._job_start: float | None = None

    def _write(self, kind: EventKind, **payload):
        t = time.time()
        obj = {"t": round(t, 6), "kind": kind, **payload}
        if self._job_start is not None:
            obj["dt"] = round(t - self._job_start, 3)
        with open(self._path, "a") as f:
            f.write(json.dumps(obj) + "\n")
        return obj

    def job_start(self, method: str, n_assays: int):
        self._job_start = time.time()
        return self._write("job_start", method=method, n_assays=n_assays)

    def model_loaded(self, model_id: str):
        return self._write("model_loaded", model_id=model_id)

    def assay_start(self, assay_id: str, L: int, n_variants: int):
        return self._write("assay_start", assay_id=assay_id, L=L, n_variants=n_variants)

    def assay_done(self, assay_id: str, L: int, wall_s: float, spearman_rho: float):
        return self._write(
            "assay_done",
            assay_id=assay_id,
            L=L,
            wall_s=round(wall_s, 3),
            spearman_rho=round(spearman_rho, 6),
        )

    def job_end(self, n_completed: int, mean_rho: float):
        return self._write("job_end", n_completed=n_completed, mean_rho=round(mean_rho, 6))
