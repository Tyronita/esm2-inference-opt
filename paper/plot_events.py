#!/usr/bin/env python3
"""
Plot performance graphs from events.jsonl produced by benchmark/events.py.

Outputs:
  paper/timeline.png   — Gantt-style per-assay timeline (wall_s vs dt_from_job_start)
  paper/beta_fit.png   — β = wall_s / L² scatter + median line

Usage:
    python paper/plot_events.py [--events results/events.jsonl] [--out paper]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_events(path: Path) -> list[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def extract_assay_spans(events: list[dict]) -> list[dict]:
    starts = {}
    spans = []
    for e in events:
        kind = e.get("kind")
        aid = e.get("assay_id", "")
        if kind == "assay_start":
            starts[aid] = e
        elif kind == "assay_done" and aid in starts:
            start_e = starts.pop(aid)
            spans.append({
                "assay_id":    aid,
                "L":           e.get("L", 0),
                "wall_s":      e.get("wall_s", 0),
                "dt_start":    start_e.get("dt", 0),
                "dt_end":      e.get("dt", 0),
                "spearman_rho": e.get("spearman_rho"),
            })
    return spans


def plot_timeline(spans: list[dict], out_dir: Path):
    if not spans:
        return
    fig, ax = plt.subplots(figsize=(10, max(3, len(spans) * 0.4)))
    for i, s in enumerate(spans):
        label = f"{s['assay_id'][:30]}  L={s['L']}"
        rho   = s.get("spearman_rho")
        color = plt.cm.RdYlGn(max(0, min(1, (rho + 1) / 2))) if rho is not None else "steelblue"
        ax.barh(i, s["dt_end"] - s["dt_start"], left=s["dt_start"],
                color=color, edgecolor="white", linewidth=0.5)
        ax.text(s["dt_end"] + 0.5, i,
                f"ρ={rho:+.3f}" if rho is not None else "", va="center", fontsize=7)
    ax.set_yticks(range(len(spans)))
    ax.set_yticklabels([s["assay_id"][:35] for s in spans], fontsize=7)
    ax.set_xlabel("Time from job start (s)")
    ax.set_title("Per-assay timing (colour = Spearman ρ, green=high)")
    plt.tight_layout()
    out = out_dir / "timeline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_beta_fit(spans: list[dict], out_dir: Path):
    spans = [s for s in spans if s["L"] > 0 and s["wall_s"] > 0]
    if not spans:
        return
    Ls   = np.array([s["L"]      for s in spans])
    ws   = np.array([s["wall_s"] for s in spans])
    betas = ws / (Ls ** 2)

    beta_hat = float(np.median(betas))

    L_plot = np.linspace(min(Ls) * 0.8, max(Ls) * 1.2, 200)
    y_fit  = beta_hat * L_plot ** 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.scatter(Ls, ws, s=40, zorder=3, label="measured")
    ax.plot(L_plot, y_fit, "r--", label=f"β̂={beta_hat:.2e} s/AA²")
    ax.set_xlabel("Sequence length L (AA)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("t(L) = β · L²")
    ax.legend()

    ax = axes[1]
    ax.scatter(Ls, betas, s=40)
    ax.axhline(beta_hat, color="r", linestyle="--", label=f"median β̂={beta_hat:.2e}")
    ax.set_xlabel("Sequence length L (AA)")
    ax.set_ylabel("β = wall_s / L² (s/AA²)")
    ax.set_title("β per assay")
    ax.legend()

    plt.suptitle("Empirical timing model: β × L²")
    plt.tight_layout()
    out = out_dir / "beta_fit.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="results/events.jsonl")
    ap.add_argument("--out",    default="paper")
    args = ap.parse_args()

    events_path = Path(args.events)
    out_dir     = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not events_path.exists():
        print(f"No events file at {events_path} — nothing to plot.")
        return

    events = load_events(events_path)
    spans  = extract_assay_spans(events)
    print(f"Found {len(events)} events, {len(spans)} completed assay spans.")

    plot_timeline(spans, out_dir)
    plot_beta_fit(spans, out_dir)


if __name__ == "__main__":
    main()
