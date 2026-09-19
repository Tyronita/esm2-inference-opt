#!/usr/bin/env python3
"""
Append-only project story generator.

Every commit appends one entry to STORY.md summarising what changed,
why, the integrity snapshot, and how to undo it.

Usage:
    python scripts/story.py              # append entry for HEAD
    python scripts/story.py --backfill   # write entries for all commits (safe: skips [story] commits)
    python scripts/story.py --check      # print what would be written, don't write
"""

import hashlib
import subprocess
import sys
from pathlib import Path

STORY_FILE  = Path(__file__).parent.parent / "STORY.md"
REPO_ROOT   = Path(__file__).parent.parent
PINNED_SHA  = "2b369911bb5b4b0dda914521b9475cad1656b2ac"
PINNED_PG   = "144fe22b07dfaeec2b366f2346203a9838a55b4c"


# ── git helpers ───────────────────────────────────────────────────────────────

def git(*args) -> str:
    return subprocess.check_output(["git"] + list(args), cwd=REPO_ROOT, text=True).strip()


def all_commits() -> list[dict]:
    """Return all commits oldest-first, excluding [story] commits."""
    raw = git("log", "--reverse", "--format=%H|%h|%an|%ai|%s")
    out = []
    for line in raw.strip().splitlines():
        sha, short, author, ts, msg = line.split("|", 4)
        if msg.startswith("[story]"):
            continue
        out.append({"sha": sha, "short": short, "author": author, "ts": ts, "msg": msg})
    return out


def files_changed(sha: str) -> tuple[list[str], str]:
    """Return (list of changed files, summary line like '3 files changed, +120 -4')."""
    try:
        parent = git("rev-parse", f"{sha}~1")
        names  = git("diff", "--name-only", f"{sha}~1", sha).strip().splitlines()
        stat   = git("diff", "--shortstat", f"{sha}~1", sha).strip()
    except subprocess.CalledProcessError:
        names = git("diff-tree", "--no-commit-id", "-r", "--name-only", sha).strip().splitlines()
        stat  = "initial commit"
    return [n for n in names if n], stat


def already_in_story(sha: str) -> bool:
    if not STORY_FILE.exists():
        return False
    return sha[:7] in STORY_FILE.read_text()


# ── integrity snapshot ────────────────────────────────────────────────────────

def merkle_root() -> str:
    inst = REPO_ROOT / ".venv" / "lib" / "python3.11" / "site-packages" / "esm"
    if not inst.exists():
        return "not-installed"
    hashes = {}
    for p in sorted(inst.rglob("*.py")):
        if "__pycache__" not in p.parts:
            hashes[str(p.relative_to(inst))] = hashlib.sha256(p.read_bytes()).hexdigest()
    blob = "\n".join(f"{k}:{v}" for k, v in sorted(hashes.items())).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def submodule_sha(path: str) -> str:
    try:
        return git("-C", path, "rev-parse", "HEAD")[:8]
    except Exception:
        return "unknown"


# ── narrative summaries ───────────────────────────────────────────────────────
# Each commit message maps to a human-readable story beat.
# For new commits, the message IS the summary — cleaned up.

def narrative(msg: str, files: list[str]) -> str:
    """One-sentence story beat derived from commit message."""
    m = msg.lower()
    if "initial commit" in m:
        return (
            "Niall started the project — reproduce ProteinGym's published ρ=0.414 ± 0.012 "
            "for ESM-2 650M masked-marginals on 217 DMS assays, then beat it."
        )
    if "submodule" in m or "pin" in m:
        return (
            "Both source implementations pinned as git submodules with exact SHAs: "
            "fair-esm @ 2b369911 (the commit that produced ρ=0.414) and ProteinGym @ 144fe22b. "
            "Reproducibility gate added: fair-esm fp32 vs transformers fp16 on SNCA, ρ must be ≥ 0.999."
        )
    if "hardware" in m or "a100" in m:
        return (
            "Corrected all entrypoints to target A100 40GB — the GPU used in the original paper. "
            "MPS backend kept for local smoke tests only."
        )
    if "ablation" in m or "6 scoring" in m or "comprehensive" in m:
        return (
            "Implemented all 6 scoring methods and 10 evaluation metrics. "
            "wt_marginals (1 pass), masked_marginals (L passes), pseudo_ppl, "
            "batched_masked B=8/32, simple_ofs. "
            "Metrics: Spearman, Pearson, Kendall, NDCG@10/50, top-k recall, AUC, MCC."
        )
    if "smoke" in m or "44/44" in m:
        return (
            "Local smoke test written and passes 44/44 checks on M3 MPS using ESM-2 8M as proxy. "
            "Confirms all 6 scoring methods, all metrics, fair-esm load, "
            "and fp16 vs fp32 precision gate (ρ=+1.000) before spending on Modal."
        )
    if "streaming" in m or "empirical" in m or "profiler" in m:
        return (
            "Per-assay streaming JSONL added: every assay logs immediately, not batch at end. "
            "RunTimer tracks TTFT, fits β̂ = median(wall_s/L²) live, prints ETA after each assay. "
            "profile_snca entrypoint added for Chrome tracing."
        )
    if "append-only" in m or "pull_results" in m:
        return (
            "results/log.md made append-only with timestamps. "
            "pull_results Modal entrypoint merges volume log → local log without overwriting."
        )
    if "mac" in m or "timing" in m or "rationale" in m:
        return (
            "Added M3 MPS timing data (SNCA 16.5s, LRRK2 85.8min), "
            "Modal vs local decision table, numbered history, "
            "and all ESM-2 model configs (8M→15B) from the paper. "
            "Corrected published baseline: 0.44 → 0.414 ± 0.012."
        )
    if "timing model" in m:
        return (
            "Per-assay streaming JSONL added: every assay logs immediately, not batch at end. "
            "RunTimer tracks TTFT, fits β̂ = median(wall_s/L²) live, prints ETA after each assay. "
            "profile_snca entrypoint added for Chrome tracing."
        )
    if "git" in m and "modal" in m:
        return (
            "Fixed Modal image: debian_slim has no git, so "
            "fair-esm @ SHA pip install was failing. "
            "Added apt-get install git as first image step."
        )
    if "dms zip" in m or "url" in m:
        return (
            "Fixed 404: GitHub releases URL for ProteinGym DMS data was removed. "
            "Corrected to marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/."
        )
    if "assay id" in m or "beta_hat" in m or "crash" in m:
        return (
            "Fixed wrong PD assay IDs (LRRK2 and GBA not in ProteinGym 217-assay set). "
            "Replaced with SYUA (SNCA) + PRKN + BRCA1 + ZIKV for length diversity. "
            "Fixed beta_hat None crash in format string."
        )
    if "latex" in m or "event" in m or "profiling" in m:
        return (
            "LaTeX reproducibility report added (paper/report.tex) with method math, "
            "architecture table, related papers, empirical timing model. "
            "EventLog class added (benchmark/events.py): append-only JSONL of "
            "job_start, model_loaded, assay_start, assay_done events for timeline plots."
        )
    # default: clean up the commit message
    return msg.rstrip(".")


# ── entry builder ─────────────────────────────────────────────────────────────

def build_entry(commit: dict) -> str:
    sha, short, author, ts, msg = (
        commit["sha"], commit["short"], commit["author"], commit["ts"], commit["msg"]
    )
    date       = ts[:10]
    names, stat_summary = files_changed(sha)

    files_str = "  ".join(f"`{f}`" for f in names[:6])
    if len(names) > 6:
        files_str += f"  _{len(names) - 6} more_"

    story_beat = narrative(msg, names)
    merkle     = merkle_root()
    esm_sha    = submodule_sha(str(REPO_ROOT / "refs" / "fair-esm"))
    pg_sha     = submodule_sha(str(REPO_ROOT / "refs" / "ProteinGym"))

    # cleanup: simple revert instruction
    cleanup = f"`git revert {short} --no-edit`"
    if "initial commit" in msg.lower():
        cleanup = "`git rm -rf . && git commit -m 'wipe'`  _(nuclear option)_"

    return f"""\n---\n
### {date} · `{short}` · {author}

**{msg}**

{story_beat}

| | |
|---|---|
| SHA | `{sha}` |
| Changed | {stat_summary if stat_summary else '—'} |
| Files | {files_str if files_str else '—'} |
| Integrity | Merkle `{merkle}` · fair-esm@`{esm_sha}` · ProteinGym@`{pg_sha}` |
| Undo | {cleanup} |
"""


# ── STORY.md header ───────────────────────────────────────────────────────────

HEADER = """\
# ESM-2 Inference Optimisation — Project Story

> **Append-only.** Every commit auto-appends one entry via `.githooks/post-commit`.
> Source code (fair-esm, ProteinGym) is never modified — only pinned as submodules.
> Integrity: `python benchmark/verify_integrity.py`
> Undo any commit: `git revert <sha> --no-edit`

Target: reproduce then beat **ρ = 0.414 ± 0.012** (ESM-2 650M, masked_marginals, 217 ProteinGym assays).
Source: Notin et al. NeurIPS 2023 / Meier et al. NeurIPS 2021.
"""


# ── main ──────────────────────────────────────────────────────────────────────

def append_head(dry_run: bool = False):
    sha   = git("log", "--format=%H", "-1")
    short = git("log", "--format=%h", "-1")
    msg   = git("log", "--format=%s", "-1")

    if msg.startswith("[story]"):
        return  # skip story commits

    commit = {
        "sha": sha, "short": short,
        "author": git("log", "--format=%an", "-1"),
        "ts": git("log", "--format=%ai", "-1"),
        "msg": msg,
    }
    entry = build_entry(commit)

    if dry_run:
        print(entry)
        return

    with open(STORY_FILE, "a") as f:
        f.write(entry)
    print(f"  [story] appended {short}: {msg[:60]}")


def backfill(dry_run: bool = False):
    commits = all_commits()

    if not STORY_FILE.exists() or STORY_FILE.stat().st_size < 100:
        if not dry_run:
            with open(STORY_FILE, "w") as f:
                f.write(HEADER)

    for commit in commits:
        if already_in_story(commit["short"]):
            print(f"  skip  {commit['short']}  (already in story)")
            continue
        entry = build_entry(commit)
        if dry_run:
            print(entry)
        else:
            with open(STORY_FILE, "a") as f:
                f.write(entry)
            print(f"  wrote {commit['short']}  {commit['msg'][:60]}")


def append_results(dry_run: bool = False):
    """Append new method run entries from results/*_stream.jsonl to STORY.md."""
    import json as _json

    results_dir = REPO_ROOT / "results"
    jsonl_files = sorted(results_dir.glob("*_stream.jsonl"))
    if not jsonl_files:
        print("  [story] no *_stream.jsonl files found")
        return

    # Read already-logged run IDs (assay+method+ts combos) from STORY.md
    already = set()
    if STORY_FILE.exists():
        body = STORY_FILE.read_text()
        # look for run markers we write below
        for m in __import__("re").findall(r"\[run:([^\]]+)\]", body):
            already.add(m)

    new_entries = 0
    for jsonl_path in jsonl_files:
        method = jsonl_path.stem.replace("_stream", "").replace("pd_proteins_", "")
        rows = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(_json.loads(line))
                except _json.JSONDecodeError:
                    pass

        if not rows:
            continue

        # Group rows by method
        by_method: dict[str, list] = {}
        for row in rows:
            m = row.get("method", method)
            by_method.setdefault(m, []).append(row)

        for m_name, m_rows in sorted(by_method.items()):
            # build a stable ID from method + assay IDs
            assay_ids = [r.get("assay_id", "?") for r in m_rows]
            run_id = f"{m_name}:{','.join(sorted(assay_ids))}"
            if run_id in already:
                continue

            gpu    = m_rows[0].get("gpu", "A100")
            ts     = m_rows[0].get("ts", "")[:10]
            n      = len(m_rows)
            rhos   = [r.get("spearman_rho") for r in m_rows if r.get("spearman_rho") is not None]
            mean_rho = sum(rhos) / len(rhos) if rhos else None
            total_wall = sum(r.get("wall_s", 0) for r in m_rows)

            # Per-assay lines
            rows_md = ""
            for r in m_rows:
                rho   = r.get("spearman_rho")
                rho_s = f"{rho:+.4f}" if rho is not None else "—"
                beta  = r.get("beta")
                beta_s = f"{beta:.2e}" if beta is not None else "—"
                rows_md += (
                    f"| {r.get('assay_id','?'):<45} "
                    f"| L={r.get('sequence_length','?'):<5} "
                    f"| N={r.get('n_variants','?'):<6} "
                    f"| ρ={rho_s} "
                    f"| β={beta_s} "
                    f"| {r.get('wall_s','?'):.1f}s |\n"
                    if isinstance(r.get("wall_s"), float)
                    else f"| {r.get('assay_id','?')} | — | — | ρ={rho_s} | β={beta_s} | — |\n"
                )

            mean_s = f"{mean_rho:+.4f}" if mean_rho is not None else "—"
            merkle = merkle_root()
            esm_s  = submodule_sha(str(REPO_ROOT / "refs" / "fair-esm"))

            entry = f"""
---

### {ts} · run · `{m_name}` on {gpu}

<!-- [run:{run_id}] -->

**Method run: `{m_name}` — {n} assays — mean ρ = {mean_s}**

Wall total: {total_wall:.1f}s · Hardware: {gpu} · Integrity: Merkle `{merkle}` · fair-esm@`{esm_s}`

| Assay | L | N | ρ | β (s/AA²) | Wall |
|---|---|---|---|---|---|
{rows_md}
"""
            if dry_run:
                print(entry)
            else:
                with open(STORY_FILE, "a") as f:
                    f.write(entry)
                new_entries += 1

    if not dry_run:
        print(f"  [story] appended {new_entries} method run entries")


if __name__ == "__main__":
    dry = "--check" in sys.argv
    if "--backfill" in sys.argv:
        backfill(dry_run=dry)
    elif "--append-results" in sys.argv:
        append_results(dry_run=dry)
    else:
        append_head(dry_run=dry)
