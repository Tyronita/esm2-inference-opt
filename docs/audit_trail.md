# Audit Trail — ESM-2 ProteinGym Benchmark

_Claude Code session log. Sensitive values redacted: emails → `[REDACTED]`, tokens → `[TOKEN]`._

---

## Project Goal

Reproduce Table 1 of Notin et al. NeurIPS 2023 (ESM-2 650M, 217 substitution assays) using the original Meta Research `fair-esm` implementation, verified against a pinned SHA. Measure A100 wall-clock timing per assay as a secondary deliverable. Push results to HuggingFace dataset `EvanOLeary/esm2-proteinGym-benchmark`.

**Published targets (Notin et al. 2023, ESM-2 650M, 217 assays):**

| Method | Published ρ |
|---|---|
| `wt_marginals` | 0.430 |
| `masked_marginals` | 0.440 |
| `pseudo_ppl` | 0.440 |

---

## Two-Track Architecture

### Track A — Reference (fp32)
- **Source**: `refs/fair-esm @ 2b369911bb5b4b0dda914521b9475cad1656b2ac`
- **API**: `model(tokens)["logits"]`, `alphabet.get_batch_converter()`, `alphabet.get_idx(aa)`, `alphabet.mask_idx`
- **Dtype**: `torch.float32`
- **Used for**: All benchmark runs (`ablate_ref_proteingym`)

### Track B — HuggingFace (fp16)
- **Source**: `transformers==4.44.0`, `facebook/esm2_t33_650M_UR50D`
- **API**: `model(input_ids=...).logits`, `tokenizer.mask_token_id`
- **Dtype**: `torch.float16`
- **Used for**: Consistency gate (`check_consistency.py`) — not used in canonical benchmark runs
- **Status**: Commented out of scoring paths pending SOTA optimisation work

### Bridge
`check_consistency.py` scores 200 SNCA variants with both tracks and asserts Spearman ρ ≥ 0.999.
**Result**: ρ = +0.999973 — PASS.

---

## Integrity Verification (`benchmark/verify_integrity.py`)

### Track A — fair-esm Merkle
- 31 `.py` files under `refs/fair-esm/esm/` verified against installed `esm` package
- Merkle root: `5e31a69554e4cd91`
- **Status**: 31/31 CLEAN

### Track B — transformers SHA256 pins
| File | SHA256 |
|---|---|
| `models/esm/modeling_esm.py` | `03e3784d40a04802164f62ccc611b8066f7fee71cf3a661c6582f1889ac68c1d` |
| `models/esm/configuration_esm.py` | `04122d1002da3e3da69b75dfb0c828bb0c2ea008a1abfb84d63799d3ca7fa3b9` |
| `models/esm/tokenization_esm.py` | `1c3e895f0777de84b0300beeb206802a96fe9053db234a4f5f3dd394bc92077b` |
- Merkle: `14dd4e9ba2393067`
- **Status**: 3/3 CLEAN

---

## Decisions and Fixes (Chronological)

### 1. Integrity gap — Track B not verified (fixed)
**Problem**: `verify_integrity.py` only Merkle-checked Track A (fair-esm). The transformers production files had no integrity check.  
**Fix**: Added SHA256 pinning for the 3 transformers ESM files. Both tracks now verified independently.  
**Commit**: `b2ba2b0`

### 2. Reference times with commit provenance
**Decision**: Canonical timing table (`results/reference_times.md`, `paper/reference_times.tex`) attaches two commit IDs per row — code commit at run launch and data commit — so every timing result is reproducible to an exact source state.  
**Code commit at run**: `fba469b` | **Data commit**: `e3567c1`  
**Commit**: `8926169`

### 3. Track A scoring implementations (`scoring/ref/`)
**Problem**: All scoring code used the Track B (transformers) API. No Track A implementation existed.  
**Fix**: New `scoring/ref/` package implementing `masked_marginals`, `wt_marginals`, `pseudo_ppl` using the fair-esm alphabet API (`alphabet.get_batch_converter()`, `alphabet.get_idx()`, `alphabet.mask_idx`). Windowed variants for sequences > 1022 tokens.  
Track B scoring files (`scoring/*.py`) marked with `# TRACK B — STATUS: commented out for benchmark runs`.

### 4. N=5 repeated-run harness (`benchmark/repeated_run.py`)
**Decision**: N warmup passes (CUDA cold-start) + N timed passes each measuring:
- `time.perf_counter()` wall clock
- `torch.cuda.Event` device-side timing (ms)
- `torch.cuda.max_memory_allocated()` peak VRAM (MB)

Returns `wall_mean`, `wall_std`, `cuda_ms_mean`, `cuda_ms_std`, `peak_mem_mb`, per-run arrays.

### 5. DMS data path mismatch — 0/217 assays ran (critical fix)
**Root cause**: `fetch_dms_data` extracted files into `dms_files/` via `extractall()` and returned a `dict` keyed by stem. The reference CSV `DMS_filename` column contains full filenames like `A0A140D2T1_ZIKV_Sourisseau_2019.csv` — these were never matched.  
**Fix**: Extract flat into `DMS_substitutions/` (matching the `DMS_filename` column), return `Path`. All callers updated: `dms_dir = fetch_dms_data(...)`, `dms_path = dms_dir / ref_row["DMS_filename"]`.  
**Commit**: `576ee5b`

### 6. PD cherry-picking bias removed
**Problem**: `compare_tracks` was importing and using `PD_ASSAY_IDS` to filter assays, which would have biased results toward Parkinson's-related proteins.  
**Fix**: `PD_ASSAY_IDS` explicitly **not** imported in `compare_tracks`. All 217 assays iterated from `reference.iterrows()` with no filter.  
Comment in code: `# NOTE: PD_ASSAY_IDS intentionally NOT imported — no cherry-picking`

### 7. Budget crisis — stopped expensive run
**Situation**: `compare_tracks` (Track A + B × 3 methods × 217 assays × N=5 repeats) was launched. Cost estimate: ~$274. Remaining budget: ~$26.  
**Decision**: Stop run, switch to lean version: Track A only, N=1 warmup + N=1 timed. Cost ~$4, ~3.6 hr.  
**Action**: `modal app stop ap-pdluRf9eTGbYqeluF5pfbm -y`

### 8. Modal type annotation fix
**Problem**: `methods: list[str] | None` in `ablate_ref_proteingym` and `assay_ids: list[str] | None` in `run_method` caused Modal CLI parse error: `Parameter has unparseable annotation`.  
**Fix**: Changed to `methods: str = ""` (comma-delimited) and `assay_ids: str = ""` with `.split(",")` parsing inside the function.

### 9. torch.profiler API compat fix
**Problem**: `FunctionEventAvg` in torch 2.4 no longer exposes `.cuda_time_total` directly, causing `AttributeError` on the first profiled assay.  
**Fix**: `_cuda_us()` shim in `benchmark/profiler.py` tries `cuda_time_total`, `self_cuda_time_total`, `device_time_total` in order.

---

## Scoring Methods

| Method | Passes | Published ρ | Notes |
|---|---|---|---|
| `wt_marginals` | 1 | 0.430 | Self-info leakage: model sees position being scored |
| `masked_marginals` | L | 0.440 | Canonical ProteinGym baseline; position isolated from context |
| `pseudo_ppl` | L (mutant) | 0.440 | Identical to masked_marginals for single-site variants |
| `batched_masked_b8` | ceil(L/8) | — | Approximation; 8 positions masked simultaneously |
| `batched_masked_b32` | ceil(L/32) | — | Larger approximation error, faster |
| `simple_ofs` | 1 | — | WARNING: functionally equivalent to wt_marginals. True OFS (Kantroo et al. 2024) requires trained MLP heads not included here |

---

## Benchmark Infrastructure

### Modal (`modal_app.py`)
- **Image**: `debian_slim` + PyTorch 2.4 (CUDA 12.1) + fair-esm @ `2b369911` + transformers 4.44.0
- **Hardware**: A100-SXM4-40GB
- **Volume**: `esm2-weights` (model weights + results cache)
- **Key entrypoints**:
  - `ablate_ref_proteingym` — canonical benchmark (Track A, all 217 assays)
  - `compare_tracks` — Track A vs Track B side-by-side (budget: ~$25–35)
  - `check_consistency` — ρ gate between tracks
  - `pull_ref_results` — pull stream JSONL from volume

### Streaming results
Results stream to `results/ref_proteingym_stream.jsonl` (one row per assay × method, append-only). Every row committed to volume immediately — resume-safe if the job is interrupted.

### HuggingFace dataset
`EvanOLeary/esm2-proteinGym-benchmark` — pushed via `scripts/push_hf_dataset.py` after run completes.

---

## Run Log

| Date (UTC) | App ID | Command | Outcome |
|---|---|---|---|
| 2026-09-20 00:17 | `ap-KSBjAQev7TRUNmbh9lqt9I` | `compare_tracks` N=5 | Stopped — estimated $274, over budget |
| 2026-09-20 00:32 | `ap-xXzXqtm5qnhiZUzZr4P6FL` | `ablate_ref_proteingym` | Stopped — Modal type annotation error |
| 2026-09-20 00:34 | `ap-UEDRrqzdmsx4TJRYlslUjV` | `ablate_ref_proteingym --n-timed 1 --n-warmup 1` | **Running** — Track A fp32, 217 assays, ~$4 |

### Progress snapshot (pulled ~00:50 UTC)
```
wt_marginals       73/217  (33.6%)  mean ρ=+0.4405  ← in progress
masked_marginals    0/217   (0.0%)  queued
pseudo_ppl          0/217   (0.0%)  queued
```

---

## File Map

```
benchmark/
  events.py           — EventLog: per-assay start/done JSONL events
  metrics.py          — Spearman, Pearson, NDCG, AUC, MCC, top-k recall
  profiler.py         — torch.profiler context manager + extract_summary
  proteingym.py       — fetch_reference(), fetch_dms_data(), run_benchmark()
  repeated_run.py     — N warmup + N timed passes, wall + CUDA timing
  timing.py           — RunTimer: TTFT, β̂, ETA
  verify_integrity.py — Track A Merkle + Track B SHA256 verifier

scoring/
  masked_marginals.py — Track B (transformers fp16) — commented out
  wt_marginals.py     — Track B (transformers fp16) — commented out
  pseudo_ppl.py       — Track B (transformers fp16) — commented out
  batched_masked.py   — Track B approximation
  simple_ofs.py       — naive single-pass (NOT Kantroo 2024 OFS)
  registry.py         — ScoringMethod dataclass + get_registry()
  ref/
    masked_marginals.py — Track A (fair-esm fp32)
    wt_marginals.py     — Track A (fair-esm fp32), windowed for L>1022
    pseudo_ppl.py       — Track A (fair-esm fp32)
    registry.py         — Track A registry with published_rho

scripts/
  push_hf_dataset.py  — push ref_proteingym_stream.jsonl to HF Hub
  reference_times.py  — generate reference_times.md + .tex with commit IDs

paper/
  report.tex          — two-track architecture, integrity table, §Results
  reference_times.tex — canonical timing table (generated)

results/
  reference_times.md         — 12-row timing reference with commit provenance
  ref_proteingym_stream.jsonl — live benchmark results (pulled from volume)

refs/
  fair-esm/   @ 2b369911  — Track A source
  ProteinGym/ @ 144fe22b  — reference CSV + DMS data

modal_app.py          — all Modal entrypoints
check_consistency.py  — ρ bridge between Track A and Track B
```

---

## Pending

1. **Benchmark complete** — wait for `ap-UEDRrqzdmsx4TJRYlslUjV` to finish (~3hr remaining)
2. **Pull results** — `modal run modal_app.py::pull_ref_results`
3. **Verify ρ matches published** — wt=0.430, mm=0.440, ppl=0.440
4. **Push to HuggingFace** — `python scripts/push_hf_dataset.py`
5. **Track B comparison run** — needs budget top-up; ~$25–35 on A100

---

_Generated by Claude Code. Repository: `esm2-inference-opt`._
