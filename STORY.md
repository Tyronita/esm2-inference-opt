# ESM-2 Inference Optimisation — Project Story

> **Append-only.** Every commit auto-appends one entry via `.githooks/post-commit`.
> Source code (fair-esm, ProteinGym) is never modified — only pinned as submodules.
> Integrity: `python benchmark/verify_integrity.py`
> Undo any commit: `git revert <sha> --no-edit`

Target: reproduce then beat **ρ = 0.414 ± 0.012** (ESM-2 650M, masked_marginals, 217 ProteinGym assays).
Source: Notin et al. NeurIPS 2023 / Meier et al. NeurIPS 2021.

---

### 2026-09-19 · `a144c44` · nialloleary

**Initial commit: ESM-2 650M inference optimisation repo**

Niall started the project — reproduce ProteinGym's published ρ=0.414 ± 0.012 for ESM-2 650M masked-marginals on 217 DMS assays, then beat it.

| | |
|---|---|
| SHA | `a144c442064631911b008d14c564ca59571294b2` |
| Changed | initial commit |
| Files | — |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git rm -rf . && git commit -m 'wipe'`  _(nuclear option)_ |

---

### 2026-09-19 · `5e7a7ac` · nialloleary

**Fix hardware to A100 40GB across all entrypoints**

Corrected all entrypoints to target A100 40GB — the GPU used in the original paper. MPS backend kept for local smoke tests only.

| | |
|---|---|
| SHA | `5e7a7ac8838a4046a41ec168c3ebb93355143e54` |
| Changed | 1 file changed, 10 insertions(+), 4 deletions(-) |
| Files | `modal_app.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 5e7a7ac --no-edit` |

---

### 2026-09-19 · `1596eb5` · nialloleary

**feat: comprehensive ablation — all 6 scoring methods × all ProteinGym metrics**

Implemented all 6 scoring methods and 10 evaluation metrics. wt_marginals (1 pass), masked_marginals (L passes), pseudo_ppl, batched_masked B=8/32, simple_ofs. Metrics: Spearman, Pearson, Kendall, NDCG@10/50, top-k recall, AUC, MCC.

| | |
|---|---|
| SHA | `1596eb594c850ad85259291e6f8a9ab387b3d671` |
| Changed | 13 files changed, 922 insertions(+), 378 deletions(-) |
| Files | `LIMITATIONS.md`  `benchmark/metrics.py`  `benchmark/proteingym.py`  `modal_app.py`  `scoring/batched_masked.py`  `scoring/masked_marginal.py`  _7 more_ |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 1596eb5 --no-edit` |

---

### 2026-09-19 · `65a887b` · nialloleary

**feat: pin all implementations as submodules + reproducibility gate**

Both source implementations pinned as git submodules with exact SHAs: fair-esm @ 2b369911 (the commit that produced ρ=0.414) and ProteinGym @ 144fe22b. Reproducibility gate added: fair-esm fp32 vs transformers fp16 on SNCA, ρ must be ≥ 0.999.

| | |
|---|---|
| SHA | `65a887b22eed703c112e57da3aa3c8c6c51657a0` |
| Changed | 8 files changed, 716 insertions(+), 6 deletions(-) |
| Files | `.gitmodules`  `benchmark/proteingym.py`  `check_consistency.py`  `modal_app.py`  `refs/ProteinGym`  `refs/README.md`  _2 more_ |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 65a887b --no-edit` |

---

### 2026-09-19 · `f113a30` · nialloleary

**feat: local smoke test — 44/44 checks pass on M3 MPS**

Local smoke test written and passes 44/44 checks on M3 MPS using ESM-2 8M as proxy. Confirms all 6 scoring methods, all metrics, fair-esm load, and fp16 vs fp32 precision gate (ρ=+1.000) before spending on Modal.

| | |
|---|---|
| SHA | `f113a300fc94e43a3b02c3ab0d4f05f719ef56c9` |
| Changed | 1 file changed, 226 insertions(+) |
| Files | `smoke_test.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert f113a30 --no-edit` |

---

### 2026-09-19 · `fda19b2` · nialloleary

**docs: add Mac M3 timing results, decision log, Modal vs local rationale**

Added M3 MPS timing data (SNCA 16.5s, LRRK2 85.8min), Modal vs local decision table, numbered history, and all ESM-2 model configs (8M→15B) from the paper. Corrected published baseline: 0.44 → 0.414 ± 0.012.

| | |
|---|---|
| SHA | `fda19b21fa25be503dd5f44aab94fe6b4282c165` |
| Changed | 1 file changed, 104 insertions(+), 44 deletions(-) |
| Files | `README.md` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert fda19b2 --no-edit` |

---

### 2026-09-19 · `84251f9` · nialloleary

**feat: append-only results log + pull_results entrypoint**

results/log.md made append-only with timestamps. pull_results Modal entrypoint merges volume log → local log without overwriting.

| | |
|---|---|
| SHA | `84251f996bb8596ced657c5ec81760b2dbd75775` |
| Changed | 2 files changed, 155 insertions(+), 2 deletions(-) |
| Files | `modal_app.py`  `results/log.md` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 84251f9 --no-edit` |

---

### 2026-09-19 · `b69ffe5` · nialloleary

**feat: streaming per-assay logging + empirical timing model + profiler**

Added M3 MPS timing data (SNCA 16.5s, LRRK2 85.8min), Modal vs local decision table, numbered history, and all ESM-2 model configs (8M→15B) from the paper. Corrected published baseline: 0.44 → 0.414 ± 0.012.

| | |
|---|---|
| SHA | `b69ffe5f51129ae6cb3d908ee950744b0ed2e72d` |
| Changed | 4 files changed, 301 insertions(+), 16 deletions(-) |
| Files | `README.md`  `benchmark/proteingym.py`  `benchmark/timing.py`  `modal_app.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert b69ffe5 --no-edit` |

---

### 2026-09-19 · `ccfa4de` · nialloleary

**fix: install git in Modal image before fair-esm SHA pip install**

Fixed Modal image: debian_slim has no git, so fair-esm @ SHA pip install was failing. Added apt-get install git as first image step.

| | |
|---|---|
| SHA | `ccfa4de10014ea10df6c772a6c3de5ac3e3219d4` |
| Changed | 1 file changed, 4 insertions(+), 2 deletions(-) |
| Files | `modal_app.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert ccfa4de --no-edit` |

---

### 2026-09-19 · `91c956a` · nialloleary

**fix: update DMS zip URL to marks.hms.harvard.edu v1.3**

Fixed 404: GitHub releases URL for ProteinGym DMS data was removed. Corrected to marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/.

| | |
|---|---|
| SHA | `91c956a77c1a7fa03ae7e1aa67c1345c179e58ca` |
| Changed | 1 file changed, 2 insertions(+), 2 deletions(-) |
| Files | `benchmark/proteingym.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 91c956a --no-edit` |

---

### 2026-09-19 · `fba469b` · nialloleary

**fix: correct PD assay IDs + beta_hat None crash**

Fixed wrong PD assay IDs (LRRK2 and GBA not in ProteinGym 217-assay set). Replaced with SYUA (SNCA) + PRKN + BRCA1 + ZIKV for length diversity. Fixed beta_hat None crash in format string.

| | |
|---|---|
| SHA | `fba469b2713cffca659f3cd1842bf8ea32680e21` |
| Changed | 1 file changed, 11 insertions(+), 5 deletions(-) |
| Files | `benchmark/proteingym.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert fba469b --no-edit` |

---

### 2026-09-19 · `7dfe175` · nialloleary

**feat: LaTeX reproducibility report + event profiling system**

LaTeX reproducibility report added (paper/report.tex) with method math, architecture table, related papers, empirical timing model. EventLog class added (benchmark/events.py): append-only JSONL of job_start, model_loaded, assay_start, assay_done events for timeline plots.

| | |
|---|---|
| SHA | `7dfe175593f5186948f053b9a8191f47790c9733` |
| Changed | 10 files changed, 739 insertions(+), 13 deletions(-) |
| Files | `.gitignore`  `benchmark/events.py`  `benchmark/proteingym.py`  `modal_app.py`  `paper/build.sh`  `paper/generate_results.py`  _4 more_ |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 7dfe175 --no-edit` |

---

### 2026-09-19 · `2d88572` · nialloleary

**feat: append-only story log + integrity hooks**

results/log.md made append-only with timestamps. pull_results Modal entrypoint merges volume log → local log without overwriting.

| | |
|---|---|
| SHA | `2d88572dc026c3990fd3ec257a8de4870d15352f` |
| Changed | 5 files changed, 651 insertions(+) |
| Files | `.githooks/post-commit`  `.githooks/pre-commit`  `STORY.md`  `benchmark/verify_integrity.py`  `scripts/story.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 2d88572 --no-edit` |

---

### 2026-09-19 · run · `masked_marginals` on NVIDIA A100-SXM4-40GB

<!-- [run:masked_marginals:A0A140D2T1_ZIKV_Sourisseau_2019,BRCA1_HUMAN_Findlay_2018,PRKN_HUMAN_Clausen_2023,SYUA_HUMAN_Newberry_2020] -->

**Method run: `masked_marginals` — 4 assays — mean ρ = +0.3424**

Wall total: 47.5s · Hardware: NVIDIA A100-SXM4-40GB · Integrity: Merkle `889c170fe685c443` · fair-esm@`2b369911`

| Assay | L | N | ρ | β (s/AA²) | Wall |
|---|---|---|---|---|---|
| SYUA_HUMAN_Newberry_2020                      | L=140   | N=2497   | ρ=+0.1372 | β=1.90e-04 | 3.7s |
| PRKN_HUMAN_Clausen_2023                       | L=465   | N=8756   | ρ=+0.5012 | β=5.94e-05 | 12.8s |
| BRCA1_HUMAN_Findlay_2018                      | L=1863  | N=1837   | ρ=+0.5154 | β=3.48e-06 | 12.1s |
| A0A140D2T1_ZIKV_Sourisseau_2019               | L=3423  | N=9576   | ρ=+0.2159 | β=1.61e-06 | 18.9s |


---

### 2026-09-19 · run · `pseudo_ppl` on NVIDIA A100-SXM4-40GB

<!-- [run:pseudo_ppl:BRCA1_HUMAN_Findlay_2018,PRKN_HUMAN_Clausen_2023,SYUA_HUMAN_Newberry_2020] -->

**Method run: `pseudo_ppl` — 3 assays — mean ρ = +0.3846**

Wall total: 371.2s · Hardware: NVIDIA A100-SXM4-40GB · Integrity: Merkle `889c170fe685c443` · fair-esm@`2b369911`

| Assay | L | N | ρ | β (s/AA²) | Wall |
|---|---|---|---|---|---|
| SYUA_HUMAN_Newberry_2020                      | L=140   | N=2497   | ρ=+0.1372 | β=3.35e-03 | 65.6s |
| PRKN_HUMAN_Clausen_2023                       | L=465   | N=8756   | ρ=+0.5012 | β=1.09e-03 | 235.1s |
| BRCA1_HUMAN_Findlay_2018                      | L=1863  | N=1837   | ρ=+0.5154 | β=2.03e-05 | 70.6s |


---

### 2026-09-19 · run · `wt_marginals` on NVIDIA A100-SXM4-40GB

<!-- [run:wt_marginals:A0A140D2T1_ZIKV_Sourisseau_2019,BRCA1_HUMAN_Findlay_2018,PRKN_HUMAN_Clausen_2023,SYUA_HUMAN_Newberry_2020,SYUA_HUMAN_Newberry_2020] -->

**Method run: `wt_marginals` — 5 assays — mean ρ = +0.2962**

Wall total: 2.9s · Hardware: NVIDIA A100-SXM4-40GB · Integrity: Merkle `889c170fe685c443` · fair-esm@`2b369911`

| Assay | L | N | ρ | β (s/AA²) | Wall |
|---|---|---|---|---|---|
| SYUA_HUMAN_Newberry_2020                      | L=140   | N=2497   | ρ=+0.1280 | β=3.05e-05 | 0.6s |
| SYUA_HUMAN_Newberry_2020                      | L=140   | N=2497   | ρ=+0.1280 | β=2.14e-05 | 0.4s |
| PRKN_HUMAN_Clausen_2023                       | L=465   | N=8756   | ρ=+0.5029 | β=1.63e-06 | 0.4s |
| BRCA1_HUMAN_Findlay_2018                      | L=1863  | N=1837   | ρ=+0.5136 | β=1.40e-07 | 0.5s |
| A0A140D2T1_ZIKV_Sourisseau_2019               | L=3423  | N=9576   | ρ=+0.2086 | β=9.00e-08 | 1.0s |


---

### 2026-09-19 · `e3567c1` · nialloleary

**feat: full LaTeX report + chat index + method results hooks**

LaTeX reproducibility report added (paper/report.tex) with method math, architecture table, related papers, empirical timing model. EventLog class added (benchmark/events.py): append-only JSONL of job_start, model_loaded, assay_start, assay_done events for timeline plots.

| | |
|---|---|
| SHA | `e3567c142f8fd688a6eaa4bb7093b2a432dc6d99` |
| Changed | 9 files changed, 1527 insertions(+), 182 deletions(-) |
| Files | `.githooks/post-commit`  `STORY.md`  `chat/index.md`  `paper/chat_index.tex`  `paper/report.tex`  `paper/results_table.tex`  _3 more_ |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert e3567c1 --no-edit` |

---

### 2026-09-19 · `b2ba2b0` · nialloleary

**fix: two-track integrity — pin transformers ESM production files by SHA256**

Both source implementations pinned as git submodules with exact SHAs: fair-esm @ 2b369911 (the commit that produced ρ=0.414) and ProteinGym @ 144fe22b. Reproducibility gate added: fair-esm fp32 vs transformers fp16 on SNCA, ρ must be ≥ 0.999.

| | |
|---|---|
| SHA | `b2ba2b0a29af33b33cb6d7e77026be0547ff4ea4` |
| Changed | 2 files changed, 202 insertions(+), 62 deletions(-) |
| Files | `benchmark/verify_integrity.py`  `paper/report.tex` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert b2ba2b0 --no-edit` |
