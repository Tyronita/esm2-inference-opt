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

---

### 2026-09-19 · `8926169` · nialloleary

**feat: reference times with commit provenance**

feat: reference times with commit provenance

| | |
|---|---|
| SHA | `89261692325e66758d6a02580164716c669b4132` |
| Changed | 4 files changed, 183 insertions(+) |
| Files | `paper/reference_times.tex`  `paper/report.tex`  `results/reference_times.md`  `scripts/reference_times.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 8926169 --no-edit` |

---

### 2026-09-19 · `0e65e6d` · nialloleary

**feat: Track A benchmark — fair-esm fp32, N=5 repeats, HF dataset push**

feat: Track A benchmark — fair-esm fp32, N=5 repeats, HF dataset push

| | |
|---|---|
| SHA | `0e65e6da977106b74e8c671671bf121229a675b8` |
| Changed | 12 files changed, 874 insertions(+), 5 deletions(-) |
| Files | `benchmark/profiler.py`  `benchmark/repeated_run.py`  `modal_app.py`  `scoring/masked_marginals.py`  `scoring/pseudo_ppl.py`  `scoring/ref/__init__.py`  _6 more_ |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 0e65e6d --no-edit` |

---

### 2026-09-20 · `dbd6015` · nialloleary

**feat: compare_tracks — all 217 assays, Track A + B, no cherry-picking**

feat: compare_tracks — all 217 assays, Track A + B, no cherry-picking

| | |
|---|---|
| SHA | `dbd6015a22a7c56b2dd8d5042784e4255d505a55` |
| Changed | 2 files changed, 240 insertions(+), 2 deletions(-) |
| Files | `modal_app.py`  `scoring/registry.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert dbd6015 --no-edit` |

---

### 2026-09-20 · `576ee5b` · nialloleary

**fix: DMS data path — fetch_dms_data extracts to DMS_substitutions/ flat dir**

fix: DMS data path — fetch_dms_data extracts to DMS_substitutions/ flat dir

| | |
|---|---|
| SHA | `576ee5b6d2f2d68bae596c4c842b11353da08e39` |
| Changed | 2 files changed, 30 insertions(+), 26 deletions(-) |
| Files | `benchmark/proteingym.py`  `modal_app.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 576ee5b --no-edit` |

---

### 2026-09-20 · `cd59e30` · nialloleary

**fix: Modal type annotations + profiler compat; add audit trail**

Per-assay streaming JSONL added: every assay logs immediately, not batch at end. RunTimer tracks TTFT, fits β̂ = median(wall_s/L²) live, prints ETA after each assay. profile_snca entrypoint added for Chrome tracing.

| | |
|---|---|
| SHA | `cd59e30d8f65cd350e7ed9a60ea99498e5f9b682` |
| Changed | 3 files changed, 238 insertions(+), 10 deletions(-) |
| Files | `benchmark/profiler.py`  `docs/audit_trail.md`  `modal_app.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert cd59e30 --no-edit` |

---

### 2026-09-20 · `8364482` · nialloleary

**docs: name Track A/B with meaningful aliases throughout**

docs: name Track A/B with meaningful aliases throughout

| | |
|---|---|
| SHA | `8364482cd142667a2502fd22283443410cfe3e7f` |
| Changed | 5 files changed, 41 insertions(+), 10 deletions(-) |
| Files | `modal_app.py`  `scoring/masked_marginals.py`  `scoring/pseudo_ppl.py`  `scoring/ref/masked_marginals.py`  `scoring/wt_marginals.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 8364482 --no-edit` |

---

### 2026-09-20 · `e413132` · nialloleary

**refactor: rename track labels A/B → fair-esm-fp32/hf-fp16**

refactor: rename track labels A/B → fair-esm-fp32/hf-fp16

| | |
|---|---|
| SHA | `e413132ce523875f94d4010feaf21a9f2493d58a` |
| Changed | 1 file changed, 7 insertions(+), 7 deletions(-) |
| Files | `modal_app.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert e413132 --no-edit` |

---

### 2026-09-20 · `f26a73e` · nialloleary

**docs: add strategy thought piece — ESM-2 interventions under $1000 budget**

docs: add strategy thought piece — ESM-2 interventions under $1000 budget

| | |
|---|---|
| SHA | `f26a73e5dae9b007f5b62be355db4caf589aa2e2` |
| Changed | 1 file changed, 184 insertions(+) |
| Files | `docs/thought_piece_esm2_strategy.md` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert f26a73e --no-edit` |

---

### 2026-09-20 · `a017038` · nialloleary

**feat(track-c): ESMC 600M — sequential SDK + shinkaevolve batched scoring**

feat(track-c): ESMC 600M — sequential SDK + shinkaevolve batched scoring

| | |
|---|---|
| SHA | `a017038fab97eadf3ebb79d7eec0263f2cc524e8` |
| Changed | 9 files changed, 760 insertions(+), 1 deletion(-) |
| Files | `.gitmodules`  `modal_app.py`  `refs/README.md`  `refs/evolutionaryscale-esm`  `refs/graph.yml`  `scoring/esmc/__init__.py`  _3 more_ |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert a017038 --no-edit` |

---

### 2026-09-20 · `b9f5d94` · nialloleary

**feat(track-c): add 3 MFU research ideas + 12-assay fast harness**

feat(track-c): add 3 MFU research ideas + 12-assay fast harness

| | |
|---|---|
| SHA | `b9f5d9465b951a40442c60e6ac8735676ff51bc4` |
| Changed | 2 files changed, 267 insertions(+), 5 deletions(-) |
| Files | `modal_app.py`  `scoring/esmc/research_ideas.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert b9f5d94 --no-edit` |

---

###  · run · `masked_marginals` on NVIDIA A100-SXM4-40GB

<!-- [run:masked_marginals:A0A140D2T1_ZIKV_Sourisseau_2019,A0A192B1T2_9HIV1_Haddox_2018,A0A1I9GEU1_NEIME_Kennouche_2019,A0A247D711_LISMN_Stadelmann_2021,A0A2Z5U3Z0_9INFA_Doud_2016,A0A2Z5U3Z0_9INFA_Wu_2014,A4D664_9INFA_Soh_2019,A4GRB6_PSEAI_Chen_2020,A4_HUMAN_Seuma_2022,AACC1_PSEAI_Dandage_2018,ACE2_HUMAN_Chan_2020,ADRB2_HUMAN_Jones_2020,AICDA_HUMAN_Gajula_2014_3cycles,AMFR_HUMAN_Tsuboyama_2023_4G3O,AMIE_PSEAE_Wrenbeck_2017,ANCSZ_Hobbs_2022,ARGR_ECOLI_Tsuboyama_2023_1AOY,B2L11_HUMAN_Dutta_2010_binding-Mcl-1,BBC1_YEAST_Tsuboyama_2023_1TG0,BCHB_CHLTE_Tsuboyama_2023_2KRU,BLAT_ECOLX_Deng_2012,BLAT_ECOLX_Firnberg_2014,BLAT_ECOLX_Jacquier_2013,BLAT_ECOLX_Stiffler_2015,BRCA1_HUMAN_Findlay_2018,BRCA2_HUMAN_Erwood_2022_HEK293T,C6KNH7_9INFA_Lee_2018,CALM1_HUMAN_Weile_2017,CAPSD_AAV2S_Sinai_2021,CAR11_HUMAN_Meitlis_2020_gof,CAR11_HUMAN_Meitlis_2020_lof,CAS9_STRP1_Spencer_2017_positive,CASP3_HUMAN_Roychowdhury_2020,CASP7_HUMAN_Roychowdhury_2020,CATR_CHLRE_Tsuboyama_2023_2AMI,CBPA2_HUMAN_Tsuboyama_2023_1O6X,CBS_HUMAN_Sun_2020,CBX4_HUMAN_Tsuboyama_2023_2K28,CCDB_ECOLI_Adkar_2012,CCDB_ECOLI_Tripathi_2016,CCR5_HUMAN_Gill_2023,CD19_HUMAN_Klesmith_2019_FMC_singles,CP2C9_HUMAN_Amorosi_2021_abundance,CP2C9_HUMAN_Amorosi_2021_activity,CSN4_MOUSE_Tsuboyama_2023_1UFM,CUE1_YEAST_Tsuboyama_2023_2MYX,D7PM05_CLYGR_Somermeyer_2022,DLG4_HUMAN_Faure_2021,DLG4_RAT_McLaughlin_2012,DN7A_SACS2_Tsuboyama_2023_1JIC,DNJA1_HUMAN_Tsuboyama_2023_2LO1,DOCK1_MOUSE_Tsuboyama_2023_2M0Y,DYR_ECOLI_Nguyen_2023,DYR_ECOLI_Thompson_2019,ENVZ_ECOLI_Ghose_2023,ENV_HV1B9_DuenasDecamp_2016,ENV_HV1BR_Haddox_2016,EPHB2_HUMAN_Tsuboyama_2023_1F0M,ERBB2_HUMAN_Elazar_2016,ESTA_BACSU_Nutschel_2020,F7YBW8_MESOW_Aakre_2015,F7YBW8_MESOW_Ding_2023,FECA_ECOLI_Tsuboyama_2023_2D1U,FKBP3_HUMAN_Tsuboyama_2023_2KFV,GAL4_YEAST_Kitzman_2015,GCN4_YEAST_Staller_2018,GDIA_HUMAN_Silverstein_2021,GFP_AEQVI_Sarkisyan_2016,GLPA_HUMAN_Elazar_2016,GRB2_HUMAN_Faure_2021,HCP_LAMBD_Tsuboyama_2023_2L6Q,HECD1_HUMAN_Tsuboyama_2023_3DKM,HEM3_HUMAN_Loggerenberg_2023,HIS7_YEAST_Pokusaeva_2019,HMDH_HUMAN_Jiang_2019,HSP82_YEAST_Cote-Hammarlof_2020_growth-H2O2,HSP82_YEAST_Flynn_2019,HSP82_YEAST_Mishra_2016,HXK4_HUMAN_Gersing_2022_activity,HXK4_HUMAN_Gersing_2023_abundance,I6TAH8_I68A0_Doud_2015,IF1_ECOLI_Kelsic_2016,ILF3_HUMAN_Tsuboyama_2023_2L33,ISDH_STAAW_Tsuboyama_2023_2LHR,KCNE1_HUMAN_Muhammad_2023_expression,KCNE1_HUMAN_Muhammad_2023_function,KCNH2_HUMAN_Kozek_2020,KCNJ2_MOUSE_Coyote-Maestas_2022_function,KCNJ2_MOUSE_Coyote-Maestas_2022_surface,KKA2_KLEPN_Melnikov_2014,LGK_LIPST_Klesmith_2015,LYAM1_HUMAN_Elazar_2016,MAFG_MOUSE_Tsuboyama_2023_1K1V,MBD11_ARATH_Tsuboyama_2023_6ACV,MET_HUMAN_Estevam_2023,MK01_HUMAN_Brenan_2016,MLAC_ECOLI_MacRae_2023,MSH2_HUMAN_Jia_2020,MTH3_HAEAE_RockahShmuel_2015,MTHR_HUMAN_Weile_2021,MYO3_YEAST_Tsuboyama_2023_2BTT,NCAP_I34A1_Doud_2015,NKX31_HUMAN_Tsuboyama_2023_2L9R,NPC1_HUMAN_Erwood_2022_HEK293T,NPC1_HUMAN_Erwood_2022_RPE1,NRAM_I33A0_Jiang_2016,NUD15_HUMAN_Suiter_2020,NUSA_ECOLI_Tsuboyama_2023_1WCL,NUSG_MYCTU_Tsuboyama_2023_2MI6,OBSCN_HUMAN_Tsuboyama_2023_1V1C,ODP2_GEOSE_Tsuboyama_2023_1W4G,OPSD_HUMAN_Wan_2019,OTC_HUMAN_Lo_2023,OTU7A_HUMAN_Tsuboyama_2023_2L2D,OXDA_RHOTO_Vanella_2023_activity,OXDA_RHOTO_Vanella_2023_expression,P53_HUMAN_Giacomelli_2018_Null_Etoposide,P53_HUMAN_Giacomelli_2018_Null_Nutlin,P53_HUMAN_Giacomelli_2018_WT_Nutlin,P53_HUMAN_Kotler_2018,P84126_THETH_Chan_2017,PABP_YEAST_Melamed_2013,PAI1_HUMAN_Huttinger_2021,PA_I34A1_Wu_2015,PHOT_CHLRE_Chen_2023,PIN1_HUMAN_Tsuboyama_2023_1I6C,PITX2_HUMAN_Tsuboyama_2023_2L7M,PKN1_HUMAN_Tsuboyama_2023_1URF,POLG_CXB3N_Mattenberger_2021,POLG_DEN26_Suphatrakul_2023,POLG_HCVJF_Qi_2014,POLG_PESV_Tsuboyama_2023_2MXD,PPARG_HUMAN_Majithia_2016,PPM1D_HUMAN_Miller_2022,PR40A_HUMAN_Tsuboyama_2023_1UZC,PRKN_HUMAN_Clausen_2023,PSAE_PICP2_Tsuboyama_2023_1PSE,PTEN_HUMAN_Matreyek_2021,PTEN_HUMAN_Mighell_2018,Q2N0S5_9HIV1_Haddox_2018,Q53Z42_HUMAN_McShan_2019_binding-TAPBPR,Q53Z42_HUMAN_McShan_2019_expression,Q59976_STRSQ_Romero_2015,Q6WV12_9MAXI_Somermeyer_2022,Q837P4_ENTFA_Meier_2023,Q837P5_ENTFA_Meier_2023,Q8WTC7_9CNID_Somermeyer_2022,R1AB_SARS2_Flynn_2022,RAD_ANTMA_Tsuboyama_2023_2CJJ,RAF1_HUMAN_Zinkus-Boltz_2019,RASH_HUMAN_Bandaru_2017,RASK_HUMAN_Weng_2022_abundance,RASK_HUMAN_Weng_2022_binding-DARPin_K55,RBP1_HUMAN_Tsuboyama_2023_2KWH,RCD1_ARATH_Tsuboyama_2023_5OAO,RCRO_LAMBD_Tsuboyama_2023_1ORC,RD23A_HUMAN_Tsuboyama_2023_1IFY,RDRP_I33A0_Li_2023,REV_HV1H2_Fernandes_2016,RFAH_ECOLI_Tsuboyama_2023_2LCL,RL20_AQUAE_Tsuboyama_2023_1GYZ,RL40A_YEAST_Mavor_2016,RL40A_YEAST_Roscoe_2013,RL40A_YEAST_Roscoe_2014,RNC_ECOLI_Weeks_2023,RPC1_BP434_Tsuboyama_2023_1R69,RPC1_LAMBD_Li_2019_high-expression,RPC1_LAMBD_Li_2019_low-expression,RS15_GEOSE_Tsuboyama_2023_1A32,S22A1_HUMAN_Yee_2023_abundance,S22A1_HUMAN_Yee_2023_activity,SAV1_MOUSE_Tsuboyama_2023_2YSB,SBI_STAAM_Tsuboyama_2023_2JVG,SC6A4_HUMAN_Young_2021,SCIN_STAAR_Tsuboyama_2023_2QFF,SCN5A_HUMAN_Glazer_2019,SDA_BACSU_Tsuboyama_2023_1PV0,SERC_HUMAN_Xie_2023,SHOC2_HUMAN_Kwon_2022,SOX30_HUMAN_Tsuboyama_2023_7JJK,SPA_STAAU_Tsuboyama_2023_1LP1,SPG1_STRSG_Olson_2014,SPG1_STRSG_Wu_2016,SPG2_STRSG_Tsuboyama_2023_5UBS,SPIKE_SARS2_Starr_2020_binding,SPIKE_SARS2_Starr_2020_expression,SPTN1_CHICK_Tsuboyama_2023_1TUD,SQSTM_MOUSE_Tsuboyama_2023_2RRU,SR43C_ARATH_Tsuboyama_2023_2N88,SRBS1_HUMAN_Tsuboyama_2023_2O2W,SRC_HUMAN_Ahler_2019,SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM,SRC_HUMAN_Nguyen_2022,SUMO1_HUMAN_Weile_2017,SYUA_HUMAN_Newberry_2020,TADBP_HUMAN_Bolognesi_2019,TAT_HV1BR_Fernandes_2016,TCRG1_MOUSE_Tsuboyama_2023_1E0L,THO1_YEAST_Tsuboyama_2023_2WQG,TNKS2_HUMAN_Tsuboyama_2023_5JRT,TPK1_HUMAN_Weile_2017,TPMT_HUMAN_Matreyek_2018,TPOR_HUMAN_Bridgford_2020,TRPC_SACS2_Chan_2017,TRPC_THEMA_Chan_2017,UBC9_HUMAN_Weile_2017,UBE4B_HUMAN_Tsuboyama_2023_3L1X,UBE4B_MOUSE_Starita_2013,UBR5_HUMAN_Tsuboyama_2023_1I2T,VG08_BPP22_Tsuboyama_2023_2GP8,VILI_CHICK_Tsuboyama_2023_1YU5,VKOR1_HUMAN_Chiasson_2020_abundance,VKOR1_HUMAN_Chiasson_2020_activity,VRPI_BPT7_Tsuboyama_2023_2WNM,YAIA_ECOLI_Tsuboyama_2023_2KVT,YAP1_HUMAN_Araya_2012,YNZC_BACSU_Tsuboyama_2023_2JVD] -->

**Method run: `masked_marginals` — 217 assays — mean ρ = +0.4391**

Wall total: 0.0s · Hardware: NVIDIA A100-SXM4-40GB · Integrity: Merkle `889c170fe685c443` · fair-esm@`2b369911`

| Assay | L | N | ρ | β (s/AA²) | Wall |
|---|---|---|---|---|---|
| A0A140D2T1_ZIKV_Sourisseau_2019 | — | — | ρ=+0.2159 | β=— | — |
| A0A192B1T2_9HIV1_Haddox_2018 | — | — | ρ=+0.0798 | β=— | — |
| A0A1I9GEU1_NEIME_Kennouche_2019 | — | — | ρ=+0.0297 | β=— | — |
| A0A247D711_LISMN_Stadelmann_2021 | — | — | ρ=+0.0661 | β=— | — |
| A0A2Z5U3Z0_9INFA_Doud_2016 | — | — | ρ=+0.5067 | β=— | — |
| A0A2Z5U3Z0_9INFA_Wu_2014 | — | — | ρ=+0.4642 | β=— | — |
| A4_HUMAN_Seuma_2022 | — | — | ρ=+0.4311 | β=— | — |
| A4D664_9INFA_Soh_2019 | — | — | ρ=+0.1488 | β=— | — |
| A4GRB6_PSEAI_Chen_2020 | — | — | ρ=+0.7382 | β=— | — |
| AACC1_PSEAI_Dandage_2018 | — | — | ρ=+0.4904 | β=— | — |
| ACE2_HUMAN_Chan_2020 | — | — | ρ=+0.2245 | β=— | — |
| ADRB2_HUMAN_Jones_2020 | — | — | ρ=+0.4926 | β=— | — |
| AICDA_HUMAN_Gajula_2014_3cycles | — | — | ρ=+0.3284 | β=— | — |
| AMFR_HUMAN_Tsuboyama_2023_4G3O | — | — | ρ=+0.2610 | β=— | — |
| AMIE_PSEAE_Wrenbeck_2017 | — | — | ρ=+0.5569 | β=— | — |
| ANCSZ_Hobbs_2022 | — | — | ρ=+0.6093 | β=— | — |
| ARGR_ECOLI_Tsuboyama_2023_1AOY | — | — | ρ=+0.4965 | β=— | — |
| B2L11_HUMAN_Dutta_2010_binding-Mcl-1 | — | — | ρ=+0.2696 | β=— | — |
| BBC1_YEAST_Tsuboyama_2023_1TG0 | — | — | ρ=+0.4792 | β=— | — |
| BCHB_CHLTE_Tsuboyama_2023_2KRU | — | — | ρ=+0.5097 | β=— | — |
| BLAT_ECOLX_Deng_2012 | — | — | ρ=+0.5281 | β=— | — |
| BLAT_ECOLX_Firnberg_2014 | — | — | ρ=+0.7371 | β=— | — |
| BLAT_ECOLX_Jacquier_2013 | — | — | ρ=+0.7038 | β=— | — |
| BLAT_ECOLX_Stiffler_2015 | — | — | ρ=+0.7315 | β=— | — |
| BRCA1_HUMAN_Findlay_2018 | — | — | ρ=+0.5153 | β=— | — |
| BRCA2_HUMAN_Erwood_2022_HEK293T | — | — | ρ=+0.5106 | β=— | — |
| C6KNH7_9INFA_Lee_2018 | — | — | ρ=+0.4830 | β=— | — |
| CALM1_HUMAN_Weile_2017 | — | — | ρ=+0.2116 | β=— | — |
| CAPSD_AAV2S_Sinai_2021 | — | — | ρ=+0.2767 | β=— | — |
| CAR11_HUMAN_Meitlis_2020_gof | — | — | ρ=+0.3272 | β=— | — |
| CAR11_HUMAN_Meitlis_2020_lof | — | — | ρ=+0.5015 | β=— | — |
| CAS9_STRP1_Spencer_2017_positive | — | — | ρ=+0.1821 | β=— | — |
| CASP3_HUMAN_Roychowdhury_2020 | — | — | ρ=+0.6375 | β=— | — |
| CASP7_HUMAN_Roychowdhury_2020 | — | — | ρ=+0.6221 | β=— | — |
| CATR_CHLRE_Tsuboyama_2023_2AMI | — | — | ρ=+0.6582 | β=— | — |
| CBPA2_HUMAN_Tsuboyama_2023_1O6X | — | — | ρ=+0.7017 | β=— | — |
| CBS_HUMAN_Sun_2020 | — | — | ρ=+0.3422 | β=— | — |
| CBX4_HUMAN_Tsuboyama_2023_2K28 | — | — | ρ=+0.6815 | β=— | — |
| CCDB_ECOLI_Adkar_2012 | — | — | ρ=+0.4659 | β=— | — |
| CCDB_ECOLI_Tripathi_2016 | — | — | ρ=+0.5115 | β=— | — |
| CCR5_HUMAN_Gill_2023 | — | — | ρ=+0.3474 | β=— | — |
| CD19_HUMAN_Klesmith_2019_FMC_singles | — | — | ρ=+0.0499 | β=— | — |
| CP2C9_HUMAN_Amorosi_2021_abundance | — | — | ρ=+0.6345 | β=— | — |
| CP2C9_HUMAN_Amorosi_2021_activity | — | — | ρ=+0.6789 | β=— | — |
| CSN4_MOUSE_Tsuboyama_2023_1UFM | — | — | ρ=+0.4838 | β=— | — |
| CUE1_YEAST_Tsuboyama_2023_2MYX | — | — | ρ=+0.5467 | β=— | — |
| D7PM05_CLYGR_Somermeyer_2022 | — | — | ρ=+0.0500 | β=— | — |
| DLG4_HUMAN_Faure_2021 | — | — | ρ=+0.5844 | β=— | — |
| DLG4_RAT_McLaughlin_2012 | — | — | ρ=+0.5428 | β=— | — |
| DN7A_SACS2_Tsuboyama_2023_1JIC | — | — | ρ=+0.3371 | β=— | — |
| DNJA1_HUMAN_Tsuboyama_2023_2LO1 | — | — | ρ=+0.8029 | β=— | — |
| DOCK1_MOUSE_Tsuboyama_2023_2M0Y | — | — | ρ=+0.5315 | β=— | — |
| DYR_ECOLI_Nguyen_2023 | — | — | ρ=+0.5442 | β=— | — |
| DYR_ECOLI_Thompson_2019 | — | — | ρ=+0.4800 | β=— | — |
| ENV_HV1B9_DuenasDecamp_2016 | — | — | ρ=+0.0122 | β=— | — |
| ENV_HV1BR_Haddox_2016 | — | — | ρ=+0.0458 | β=— | — |
| ENVZ_ECOLI_Ghose_2023 | — | — | ρ=+0.1716 | β=— | — |
| EPHB2_HUMAN_Tsuboyama_2023_1F0M | — | — | ρ=+0.8105 | β=— | — |
| ERBB2_HUMAN_Elazar_2016 | — | — | ρ=+0.4220 | β=— | — |
| ESTA_BACSU_Nutschel_2020 | — | — | ρ=+0.2995 | β=— | — |
| F7YBW8_MESOW_Ding_2023 | — | — | ρ=+0.4461 | β=— | — |
| F7YBW8_MESOW_Aakre_2015 | — | — | ρ=+0.3829 | β=— | — |
| FECA_ECOLI_Tsuboyama_2023_2D1U | — | — | ρ=+0.5237 | β=— | — |
| FKBP3_HUMAN_Tsuboyama_2023_2KFV | — | — | ρ=+0.1878 | β=— | — |
| GAL4_YEAST_Kitzman_2015 | — | — | ρ=+0.6676 | β=— | — |
| GCN4_YEAST_Staller_2018 | — | — | ρ=+0.2811 | β=— | — |
| GDIA_HUMAN_Silverstein_2021 | — | — | ρ=+0.3971 | β=— | — |
| GFP_AEQVI_Sarkisyan_2016 | — | — | ρ=+0.1078 | β=— | — |
| GLPA_HUMAN_Elazar_2016 | — | — | ρ=+0.4253 | β=— | — |
| GRB2_HUMAN_Faure_2021 | — | — | ρ=+0.6474 | β=— | — |
| HCP_LAMBD_Tsuboyama_2023_2L6Q | — | — | ρ=+0.6859 | β=— | — |
| HECD1_HUMAN_Tsuboyama_2023_3DKM | — | — | ρ=+0.3867 | β=— | — |
| HEM3_HUMAN_Loggerenberg_2023 | — | — | ρ=+0.3932 | β=— | — |
| HIS7_YEAST_Pokusaeva_2019 | — | — | ρ=+0.4113 | β=— | — |
| HMDH_HUMAN_Jiang_2019 | — | — | ρ=+0.5089 | β=— | — |
| HSP82_YEAST_Cote-Hammarlof_2020_growth-H2O2 | — | — | ρ=+0.2565 | β=— | — |
| HSP82_YEAST_Flynn_2019 | — | — | ρ=+0.2833 | β=— | — |
| HSP82_YEAST_Mishra_2016 | — | — | ρ=+0.4612 | β=— | — |
| HXK4_HUMAN_Gersing_2022_activity | — | — | ρ=+0.5166 | β=— | — |
| HXK4_HUMAN_Gersing_2023_abundance | — | — | ρ=+0.3674 | β=— | — |
| I6TAH8_I68A0_Doud_2015 | — | — | ρ=+0.0204 | β=— | — |
| IF1_ECOLI_Kelsic_2016 | — | — | ρ=+0.5987 | β=— | — |
| ILF3_HUMAN_Tsuboyama_2023_2L33 | — | — | ρ=+0.3179 | β=— | — |
| ISDH_STAAW_Tsuboyama_2023_2LHR | — | — | ρ=+0.4291 | β=— | — |
| KCNE1_HUMAN_Muhammad_2023_expression | — | — | ρ=+0.1060 | β=— | — |
| KCNE1_HUMAN_Muhammad_2023_function | — | — | ρ=+0.5247 | β=— | — |
| KCNH2_HUMAN_Kozek_2020 | — | — | ρ=+0.5112 | β=— | — |
| KCNJ2_MOUSE_Coyote-Maestas_2022_function | — | — | ρ=+0.3820 | β=— | — |
| KCNJ2_MOUSE_Coyote-Maestas_2022_surface | — | — | ρ=+0.3516 | β=— | — |
| KKA2_KLEPN_Melnikov_2014 | — | — | ρ=+0.6014 | β=— | — |
| LGK_LIPST_Klesmith_2015 | — | — | ρ=+0.5151 | β=— | — |
| LYAM1_HUMAN_Elazar_2016 | — | — | ρ=+0.3102 | β=— | — |
| MAFG_MOUSE_Tsuboyama_2023_1K1V | — | — | ρ=+0.4803 | β=— | — |
| MBD11_ARATH_Tsuboyama_2023_6ACV | — | — | ρ=+0.6987 | β=— | — |
| MET_HUMAN_Estevam_2023 | — | — | ρ=+0.5899 | β=— | — |
| MK01_HUMAN_Brenan_2016 | — | — | ρ=+0.1782 | β=— | — |
| MLAC_ECOLI_MacRae_2023 | — | — | ρ=+0.3821 | β=— | — |
| MSH2_HUMAN_Jia_2020 | — | — | ρ=+0.3395 | β=— | — |
| MTH3_HAEAE_RockahShmuel_2015 | — | — | ρ=+0.5270 | β=— | — |
| MTHR_HUMAN_Weile_2021 | — | — | ρ=+0.3475 | β=— | — |
| MYO3_YEAST_Tsuboyama_2023_2BTT | — | — | ρ=+0.5880 | β=— | — |
| NCAP_I34A1_Doud_2015 | — | — | ρ=+0.0278 | β=— | — |
| NKX31_HUMAN_Tsuboyama_2023_2L9R | — | — | ρ=+0.6511 | β=— | — |
| NPC1_HUMAN_Erwood_2022_HEK293T | — | — | ρ=+0.6946 | β=— | — |
| NPC1_HUMAN_Erwood_2022_RPE1 | — | — | ρ=+0.6604 | β=— | — |
| NRAM_I33A0_Jiang_2016 | — | — | ρ=+0.1608 | β=— | — |
| NUD15_HUMAN_Suiter_2020 | — | — | ρ=+0.5256 | β=— | — |
| NUSA_ECOLI_Tsuboyama_2023_1WCL | — | — | ρ=+0.4943 | β=— | — |
| NUSG_MYCTU_Tsuboyama_2023_2MI6 | — | — | ρ=+0.5184 | β=— | — |
| OBSCN_HUMAN_Tsuboyama_2023_1V1C | — | — | ρ=+0.7971 | β=— | — |
| ODP2_GEOSE_Tsuboyama_2023_1W4G | — | — | ρ=+0.2211 | β=— | — |
| OPSD_HUMAN_Wan_2019 | — | — | ρ=+0.5520 | β=— | — |
| OTC_HUMAN_Lo_2023 | — | — | ρ=+0.5309 | β=— | — |
| OTU7A_HUMAN_Tsuboyama_2023_2L2D | — | — | ρ=+0.3195 | β=— | — |
| OXDA_RHOTO_Vanella_2023_activity | — | — | ρ=+0.3702 | β=— | — |
| OXDA_RHOTO_Vanella_2023_expression | — | — | ρ=+0.3150 | β=— | — |
| P53_HUMAN_Giacomelli_2018_Null_Etoposide | — | — | ρ=+0.4323 | β=— | — |
| P53_HUMAN_Giacomelli_2018_Null_Nutlin | — | — | ρ=+0.4043 | β=— | — |
| P53_HUMAN_Giacomelli_2018_WT_Nutlin | — | — | ρ=+0.4002 | β=— | — |
| P53_HUMAN_Kotler_2018 | — | — | ρ=+0.6781 | β=— | — |
| P84126_THETH_Chan_2017 | — | — | ρ=+0.6047 | β=— | — |
| PA_I34A1_Wu_2015 | — | — | ρ=+0.0381 | β=— | — |
| PABP_YEAST_Melamed_2013 | — | — | ρ=+0.7158 | β=— | — |
| PAI1_HUMAN_Huttinger_2021 | — | — | ρ=+0.4479 | β=— | — |
| PHOT_CHLRE_Chen_2023 | — | — | ρ=+0.7491 | β=— | — |
| PIN1_HUMAN_Tsuboyama_2023_1I6C | — | — | ρ=+0.6699 | β=— | — |
| PITX2_HUMAN_Tsuboyama_2023_2L7M | — | — | ρ=+0.6262 | β=— | — |
| PKN1_HUMAN_Tsuboyama_2023_1URF | — | — | ρ=+0.2977 | β=— | — |
| POLG_CXB3N_Mattenberger_2021 | — | — | ρ=+0.3938 | β=— | — |
| POLG_DEN26_Suphatrakul_2023 | — | — | ρ=+0.1533 | β=— | — |
| POLG_HCVJF_Qi_2014 | — | — | ρ=+0.1307 | β=— | — |
| POLG_PESV_Tsuboyama_2023_2MXD | — | — | ρ=+0.1598 | β=— | — |
| PPARG_HUMAN_Majithia_2016 | — | — | ρ=+0.5937 | β=— | — |
| PPM1D_HUMAN_Miller_2022 | — | — | ρ=+0.6018 | β=— | — |
| PR40A_HUMAN_Tsuboyama_2023_1UZC | — | — | ρ=+0.8047 | β=— | — |
| PRKN_HUMAN_Clausen_2023 | — | — | ρ=+0.5012 | β=— | — |
| PSAE_PICP2_Tsuboyama_2023_1PSE | — | — | ρ=+0.7075 | β=— | — |
| PTEN_HUMAN_Matreyek_2021 | — | — | ρ=+0.4650 | β=— | — |
| PTEN_HUMAN_Mighell_2018 | — | — | ρ=+0.5185 | β=— | — |
| Q2N0S5_9HIV1_Haddox_2018 | — | — | ρ=+0.0442 | β=— | — |
| Q53Z42_HUMAN_McShan_2019_binding-TAPBPR | — | — | ρ=+0.3301 | β=— | — |
| Q53Z42_HUMAN_McShan_2019_expression | — | — | ρ=+0.5540 | β=— | — |
| Q59976_STRSQ_Romero_2015 | — | — | ρ=+0.5721 | β=— | — |
| Q6WV12_9MAXI_Somermeyer_2022 | — | — | ρ=+0.0096 | β=— | — |
| Q837P4_ENTFA_Meier_2023 | — | — | ρ=+0.5150 | β=— | — |
| Q837P5_ENTFA_Meier_2023 | — | — | ρ=+0.3738 | β=— | — |
| Q8WTC7_9CNID_Somermeyer_2022 | — | — | ρ=-0.0248 | β=— | — |
| R1AB_SARS2_Flynn_2022 | — | — | ρ=+0.1048 | β=— | — |
| RAD_ANTMA_Tsuboyama_2023_2CJJ | — | — | ρ=+0.4928 | β=— | — |
| RAF1_HUMAN_Zinkus-Boltz_2019 | — | — | ρ=+0.4733 | β=— | — |
| RASH_HUMAN_Bandaru_2017 | — | — | ρ=+0.4976 | β=— | — |
| RASK_HUMAN_Weng_2022_abundance | — | — | ρ=+0.2710 | β=— | — |
| RASK_HUMAN_Weng_2022_binding-DARPin_K55 | — | — | ρ=+0.6524 | β=— | — |
| RBP1_HUMAN_Tsuboyama_2023_2KWH | — | — | ρ=+0.5524 | β=— | — |
| RCD1_ARATH_Tsuboyama_2023_5OAO | — | — | ρ=+0.5049 | β=— | — |
| RCRO_LAMBD_Tsuboyama_2023_1ORC | — | — | ρ=+0.5958 | β=— | — |
| RD23A_HUMAN_Tsuboyama_2023_1IFY | — | — | ρ=+0.5127 | β=— | — |
| RDRP_I33A0_Li_2023 | — | — | ρ=+0.3218 | β=— | — |
| REV_HV1H2_Fernandes_2016 | — | — | ρ=+0.2404 | β=— | — |
| RFAH_ECOLI_Tsuboyama_2023_2LCL | — | — | ρ=+0.2991 | β=— | — |
| RL20_AQUAE_Tsuboyama_2023_1GYZ | — | — | ρ=+0.6989 | β=— | — |
| RL40A_YEAST_Mavor_2016 | — | — | ρ=+0.5179 | β=— | — |
| RL40A_YEAST_Roscoe_2013 | — | — | ρ=+0.5986 | β=— | — |
| RL40A_YEAST_Roscoe_2014 | — | — | ρ=+0.5300 | β=— | — |
| RNC_ECOLI_Weeks_2023 | — | — | ρ=+0.5962 | β=— | — |
| RPC1_BP434_Tsuboyama_2023_1R69 | — | — | ρ=+0.7084 | β=— | — |
| RPC1_LAMBD_Li_2019_high-expression | — | — | ρ=+0.5494 | β=— | — |
| RPC1_LAMBD_Li_2019_low-expression | — | — | ρ=+0.5231 | β=— | — |
| RS15_GEOSE_Tsuboyama_2023_1A32 | — | — | ρ=+0.4087 | β=— | — |
| S22A1_HUMAN_Yee_2023_abundance | — | — | ρ=+0.6264 | β=— | — |
| S22A1_HUMAN_Yee_2023_activity | — | — | ρ=+0.5752 | β=— | — |
| SAV1_MOUSE_Tsuboyama_2023_2YSB | — | — | ρ=+0.5120 | β=— | — |
| SBI_STAAM_Tsuboyama_2023_2JVG | — | — | ρ=+0.5726 | β=— | — |
| SC6A4_HUMAN_Young_2021 | — | — | ρ=+0.5426 | β=— | — |
| SCIN_STAAR_Tsuboyama_2023_2QFF | — | — | ρ=+0.3459 | β=— | — |
| SCN5A_HUMAN_Glazer_2019 | — | — | ρ=+0.1224 | β=— | — |
| SDA_BACSU_Tsuboyama_2023_1PV0 | — | — | ρ=+0.5943 | β=— | — |
| SERC_HUMAN_Xie_2023 | — | — | ρ=+0.5575 | β=— | — |
| SHOC2_HUMAN_Kwon_2022 | — | — | ρ=+0.4255 | β=— | — |
| SOX30_HUMAN_Tsuboyama_2023_7JJK | — | — | ρ=+0.3216 | β=— | — |
| SPA_STAAU_Tsuboyama_2023_1LP1 | — | — | ρ=+0.0053 | β=— | — |
| SPG1_STRSG_Olson_2014 | — | — | ρ=+0.3008 | β=— | — |
| SPG1_STRSG_Wu_2016 | — | — | ρ=+0.1283 | β=— | — |
| SPG2_STRSG_Tsuboyama_2023_5UBS | — | — | ρ=+0.4999 | β=— | — |
| SPIKE_SARS2_Starr_2020_binding | — | — | ρ=-0.0173 | β=— | — |
| SPIKE_SARS2_Starr_2020_expression | — | — | ρ=+0.0133 | β=— | — |
| SPTN1_CHICK_Tsuboyama_2023_1TUD | — | — | ρ=+0.6368 | β=— | — |
| SQSTM_MOUSE_Tsuboyama_2023_2RRU | — | — | ρ=+0.6180 | β=— | — |
| SR43C_ARATH_Tsuboyama_2023_2N88 | — | — | ρ=+0.6748 | β=— | — |
| SRBS1_HUMAN_Tsuboyama_2023_2O2W | — | — | ρ=+0.7339 | β=— | — |
| SRC_HUMAN_Ahler_2019 | — | — | ρ=+0.5551 | β=— | — |
| SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM | — | — | ρ=+0.4646 | β=— | — |
| SRC_HUMAN_Nguyen_2022 | — | — | ρ=+0.4707 | β=— | — |
| SUMO1_HUMAN_Weile_2017 | — | — | ρ=+0.5093 | β=— | — |
| SYUA_HUMAN_Newberry_2020 | — | — | ρ=+0.1373 | β=— | — |
| TADBP_HUMAN_Bolognesi_2019 | — | — | ρ=-0.1198 | β=— | — |
| TAT_HV1BR_Fernandes_2016 | — | — | ρ=+0.0174 | β=— | — |
| TCRG1_MOUSE_Tsuboyama_2023_1E0L | — | — | ρ=+0.7692 | β=— | — |
| THO1_YEAST_Tsuboyama_2023_2WQG | — | — | ρ=+0.5126 | β=— | — |
| TNKS2_HUMAN_Tsuboyama_2023_5JRT | — | — | ρ=+0.5252 | β=— | — |
| TPK1_HUMAN_Weile_2017 | — | — | ρ=+0.3386 | β=— | — |
| TPMT_HUMAN_Matreyek_2018 | — | — | ρ=+0.5387 | β=— | — |
| TPOR_HUMAN_Bridgford_2020 | — | — | ρ=+0.2899 | β=— | — |
| TRPC_SACS2_Chan_2017 | — | — | ρ=+0.6525 | β=— | — |
| TRPC_THEMA_Chan_2017 | — | — | ρ=+0.4637 | β=— | — |
| UBC9_HUMAN_Weile_2017 | — | — | ρ=+0.4726 | β=— | — |
| UBE4B_HUMAN_Tsuboyama_2023_3L1X | — | — | ρ=+0.4599 | β=— | — |
| UBE4B_MOUSE_Starita_2013 | — | — | ρ=+0.4584 | β=— | — |
| UBR5_HUMAN_Tsuboyama_2023_1I2T | — | — | ρ=+0.1969 | β=— | — |
| VG08_BPP22_Tsuboyama_2023_2GP8 | — | — | ρ=+0.6552 | β=— | — |
| VILI_CHICK_Tsuboyama_2023_1YU5 | — | — | ρ=+0.6617 | β=— | — |
| VKOR1_HUMAN_Chiasson_2020_abundance | — | — | ρ=+0.4933 | β=— | — |
| VKOR1_HUMAN_Chiasson_2020_activity | — | — | ρ=+0.4303 | β=— | — |
| VRPI_BPT7_Tsuboyama_2023_2WNM | — | — | ρ=+0.5875 | β=— | — |
| YAIA_ECOLI_Tsuboyama_2023_2KVT | — | — | ρ=+0.6245 | β=— | — |
| YAP1_HUMAN_Araya_2012 | — | — | ρ=+0.4661 | β=— | — |
| YNZC_BACSU_Tsuboyama_2023_2JVD | — | — | ρ=+0.6584 | β=— | — |


---

###  · run · `wt_marginals` on NVIDIA A100-SXM4-40GB

<!-- [run:wt_marginals:A0A140D2T1_ZIKV_Sourisseau_2019,A0A192B1T2_9HIV1_Haddox_2018,A0A1I9GEU1_NEIME_Kennouche_2019,A0A247D711_LISMN_Stadelmann_2021,A0A2Z5U3Z0_9INFA_Doud_2016,A0A2Z5U3Z0_9INFA_Wu_2014,A4D664_9INFA_Soh_2019,A4GRB6_PSEAI_Chen_2020,A4_HUMAN_Seuma_2022,AACC1_PSEAI_Dandage_2018,ACE2_HUMAN_Chan_2020,ADRB2_HUMAN_Jones_2020,AICDA_HUMAN_Gajula_2014_3cycles,AMFR_HUMAN_Tsuboyama_2023_4G3O,AMIE_PSEAE_Wrenbeck_2017,ANCSZ_Hobbs_2022,ARGR_ECOLI_Tsuboyama_2023_1AOY,B2L11_HUMAN_Dutta_2010_binding-Mcl-1,BBC1_YEAST_Tsuboyama_2023_1TG0,BCHB_CHLTE_Tsuboyama_2023_2KRU,BLAT_ECOLX_Deng_2012,BLAT_ECOLX_Firnberg_2014,BLAT_ECOLX_Jacquier_2013,BLAT_ECOLX_Stiffler_2015,BRCA1_HUMAN_Findlay_2018,BRCA2_HUMAN_Erwood_2022_HEK293T,C6KNH7_9INFA_Lee_2018,CALM1_HUMAN_Weile_2017,CAPSD_AAV2S_Sinai_2021,CAR11_HUMAN_Meitlis_2020_gof,CAR11_HUMAN_Meitlis_2020_lof,CAS9_STRP1_Spencer_2017_positive,CASP3_HUMAN_Roychowdhury_2020,CASP7_HUMAN_Roychowdhury_2020,CATR_CHLRE_Tsuboyama_2023_2AMI,CBPA2_HUMAN_Tsuboyama_2023_1O6X,CBS_HUMAN_Sun_2020,CBX4_HUMAN_Tsuboyama_2023_2K28,CCDB_ECOLI_Adkar_2012,CCDB_ECOLI_Tripathi_2016,CCR5_HUMAN_Gill_2023,CD19_HUMAN_Klesmith_2019_FMC_singles,CP2C9_HUMAN_Amorosi_2021_abundance,CP2C9_HUMAN_Amorosi_2021_activity,CSN4_MOUSE_Tsuboyama_2023_1UFM,CUE1_YEAST_Tsuboyama_2023_2MYX,D7PM05_CLYGR_Somermeyer_2022,DLG4_HUMAN_Faure_2021,DLG4_RAT_McLaughlin_2012,DN7A_SACS2_Tsuboyama_2023_1JIC,DNJA1_HUMAN_Tsuboyama_2023_2LO1,DOCK1_MOUSE_Tsuboyama_2023_2M0Y,DYR_ECOLI_Nguyen_2023,DYR_ECOLI_Thompson_2019,ENVZ_ECOLI_Ghose_2023,ENV_HV1B9_DuenasDecamp_2016,ENV_HV1BR_Haddox_2016,EPHB2_HUMAN_Tsuboyama_2023_1F0M,ERBB2_HUMAN_Elazar_2016,ESTA_BACSU_Nutschel_2020,F7YBW8_MESOW_Aakre_2015,F7YBW8_MESOW_Ding_2023,FECA_ECOLI_Tsuboyama_2023_2D1U,FKBP3_HUMAN_Tsuboyama_2023_2KFV,GAL4_YEAST_Kitzman_2015,GCN4_YEAST_Staller_2018,GDIA_HUMAN_Silverstein_2021,GFP_AEQVI_Sarkisyan_2016,GLPA_HUMAN_Elazar_2016,GRB2_HUMAN_Faure_2021,HCP_LAMBD_Tsuboyama_2023_2L6Q,HECD1_HUMAN_Tsuboyama_2023_3DKM,HEM3_HUMAN_Loggerenberg_2023,HIS7_YEAST_Pokusaeva_2019,HMDH_HUMAN_Jiang_2019,HSP82_YEAST_Cote-Hammarlof_2020_growth-H2O2,HSP82_YEAST_Flynn_2019,HSP82_YEAST_Mishra_2016,HXK4_HUMAN_Gersing_2022_activity,HXK4_HUMAN_Gersing_2023_abundance,I6TAH8_I68A0_Doud_2015,IF1_ECOLI_Kelsic_2016,ILF3_HUMAN_Tsuboyama_2023_2L33,ISDH_STAAW_Tsuboyama_2023_2LHR,KCNE1_HUMAN_Muhammad_2023_expression,KCNE1_HUMAN_Muhammad_2023_function,KCNH2_HUMAN_Kozek_2020,KCNJ2_MOUSE_Coyote-Maestas_2022_function,KCNJ2_MOUSE_Coyote-Maestas_2022_surface,KKA2_KLEPN_Melnikov_2014,LGK_LIPST_Klesmith_2015,LYAM1_HUMAN_Elazar_2016,MAFG_MOUSE_Tsuboyama_2023_1K1V,MBD11_ARATH_Tsuboyama_2023_6ACV,MET_HUMAN_Estevam_2023,MK01_HUMAN_Brenan_2016,MLAC_ECOLI_MacRae_2023,MSH2_HUMAN_Jia_2020,MTH3_HAEAE_RockahShmuel_2015,MTHR_HUMAN_Weile_2021,MYO3_YEAST_Tsuboyama_2023_2BTT,NCAP_I34A1_Doud_2015,NKX31_HUMAN_Tsuboyama_2023_2L9R,NPC1_HUMAN_Erwood_2022_HEK293T,NPC1_HUMAN_Erwood_2022_RPE1,NRAM_I33A0_Jiang_2016,NUD15_HUMAN_Suiter_2020,NUSA_ECOLI_Tsuboyama_2023_1WCL,NUSG_MYCTU_Tsuboyama_2023_2MI6,OBSCN_HUMAN_Tsuboyama_2023_1V1C,ODP2_GEOSE_Tsuboyama_2023_1W4G,OPSD_HUMAN_Wan_2019,OTC_HUMAN_Lo_2023,OTU7A_HUMAN_Tsuboyama_2023_2L2D,OXDA_RHOTO_Vanella_2023_activity,OXDA_RHOTO_Vanella_2023_expression,P53_HUMAN_Giacomelli_2018_Null_Etoposide,P53_HUMAN_Giacomelli_2018_Null_Nutlin,P53_HUMAN_Giacomelli_2018_WT_Nutlin,P53_HUMAN_Kotler_2018,P84126_THETH_Chan_2017,PABP_YEAST_Melamed_2013,PAI1_HUMAN_Huttinger_2021,PA_I34A1_Wu_2015,PHOT_CHLRE_Chen_2023,PIN1_HUMAN_Tsuboyama_2023_1I6C,PITX2_HUMAN_Tsuboyama_2023_2L7M,PKN1_HUMAN_Tsuboyama_2023_1URF,POLG_CXB3N_Mattenberger_2021,POLG_DEN26_Suphatrakul_2023,POLG_HCVJF_Qi_2014,POLG_PESV_Tsuboyama_2023_2MXD,PPARG_HUMAN_Majithia_2016,PPM1D_HUMAN_Miller_2022,PR40A_HUMAN_Tsuboyama_2023_1UZC,PRKN_HUMAN_Clausen_2023,PSAE_PICP2_Tsuboyama_2023_1PSE,PTEN_HUMAN_Matreyek_2021,PTEN_HUMAN_Mighell_2018,Q2N0S5_9HIV1_Haddox_2018,Q53Z42_HUMAN_McShan_2019_binding-TAPBPR,Q53Z42_HUMAN_McShan_2019_expression,Q59976_STRSQ_Romero_2015,Q6WV12_9MAXI_Somermeyer_2022,Q837P4_ENTFA_Meier_2023,Q837P5_ENTFA_Meier_2023,Q8WTC7_9CNID_Somermeyer_2022,R1AB_SARS2_Flynn_2022,RAD_ANTMA_Tsuboyama_2023_2CJJ,RAF1_HUMAN_Zinkus-Boltz_2019,RASH_HUMAN_Bandaru_2017,RASK_HUMAN_Weng_2022_abundance,RASK_HUMAN_Weng_2022_binding-DARPin_K55,RBP1_HUMAN_Tsuboyama_2023_2KWH,RCD1_ARATH_Tsuboyama_2023_5OAO,RCRO_LAMBD_Tsuboyama_2023_1ORC,RD23A_HUMAN_Tsuboyama_2023_1IFY,RDRP_I33A0_Li_2023,REV_HV1H2_Fernandes_2016,RFAH_ECOLI_Tsuboyama_2023_2LCL,RL20_AQUAE_Tsuboyama_2023_1GYZ,RL40A_YEAST_Mavor_2016,RL40A_YEAST_Roscoe_2013,RL40A_YEAST_Roscoe_2014,RNC_ECOLI_Weeks_2023,RPC1_BP434_Tsuboyama_2023_1R69,RPC1_LAMBD_Li_2019_high-expression,RPC1_LAMBD_Li_2019_low-expression,RS15_GEOSE_Tsuboyama_2023_1A32,S22A1_HUMAN_Yee_2023_abundance,S22A1_HUMAN_Yee_2023_activity,SAV1_MOUSE_Tsuboyama_2023_2YSB,SBI_STAAM_Tsuboyama_2023_2JVG,SC6A4_HUMAN_Young_2021,SCIN_STAAR_Tsuboyama_2023_2QFF,SCN5A_HUMAN_Glazer_2019,SDA_BACSU_Tsuboyama_2023_1PV0,SERC_HUMAN_Xie_2023,SHOC2_HUMAN_Kwon_2022,SOX30_HUMAN_Tsuboyama_2023_7JJK,SPA_STAAU_Tsuboyama_2023_1LP1,SPG1_STRSG_Olson_2014,SPG1_STRSG_Wu_2016,SPG2_STRSG_Tsuboyama_2023_5UBS,SPIKE_SARS2_Starr_2020_binding,SPIKE_SARS2_Starr_2020_expression,SPTN1_CHICK_Tsuboyama_2023_1TUD,SQSTM_MOUSE_Tsuboyama_2023_2RRU,SR43C_ARATH_Tsuboyama_2023_2N88,SRBS1_HUMAN_Tsuboyama_2023_2O2W,SRC_HUMAN_Ahler_2019,SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM,SRC_HUMAN_Nguyen_2022,SUMO1_HUMAN_Weile_2017,SYUA_HUMAN_Newberry_2020,TADBP_HUMAN_Bolognesi_2019,TAT_HV1BR_Fernandes_2016,TCRG1_MOUSE_Tsuboyama_2023_1E0L,THO1_YEAST_Tsuboyama_2023_2WQG,TNKS2_HUMAN_Tsuboyama_2023_5JRT,TPK1_HUMAN_Weile_2017,TPMT_HUMAN_Matreyek_2018,TPOR_HUMAN_Bridgford_2020,TRPC_SACS2_Chan_2017,TRPC_THEMA_Chan_2017,UBC9_HUMAN_Weile_2017,UBE4B_HUMAN_Tsuboyama_2023_3L1X,UBE4B_MOUSE_Starita_2013,UBR5_HUMAN_Tsuboyama_2023_1I2T,VG08_BPP22_Tsuboyama_2023_2GP8,VILI_CHICK_Tsuboyama_2023_1YU5,VKOR1_HUMAN_Chiasson_2020_abundance,VKOR1_HUMAN_Chiasson_2020_activity,VRPI_BPT7_Tsuboyama_2023_2WNM,YAIA_ECOLI_Tsuboyama_2023_2KVT,YAP1_HUMAN_Araya_2012,YNZC_BACSU_Tsuboyama_2023_2JVD] -->

**Method run: `wt_marginals` — 217 assays — mean ρ = +0.4441**

Wall total: 0.0s · Hardware: NVIDIA A100-SXM4-40GB · Integrity: Merkle `889c170fe685c443` · fair-esm@`2b369911`

| Assay | L | N | ρ | β (s/AA²) | Wall |
|---|---|---|---|---|---|
| A0A140D2T1_ZIKV_Sourisseau_2019 | — | — | ρ=+0.2087 | β=— | — |
| A0A192B1T2_9HIV1_Haddox_2018 | — | — | ρ=+0.0642 | β=— | — |
| A0A1I9GEU1_NEIME_Kennouche_2019 | — | — | ρ=+0.0315 | β=— | — |
| A0A247D711_LISMN_Stadelmann_2021 | — | — | ρ=+0.0656 | β=— | — |
| A0A2Z5U3Z0_9INFA_Doud_2016 | — | — | ρ=+0.4922 | β=— | — |
| A0A2Z5U3Z0_9INFA_Wu_2014 | — | — | ρ=+0.4522 | β=— | — |
| A4_HUMAN_Seuma_2022 | — | — | ρ=+0.4398 | β=— | — |
| A4D664_9INFA_Soh_2019 | — | — | ρ=+0.1370 | β=— | — |
| A4GRB6_PSEAI_Chen_2020 | — | — | ρ=+0.7281 | β=— | — |
| AACC1_PSEAI_Dandage_2018 | — | — | ρ=+0.4762 | β=— | — |
| ACE2_HUMAN_Chan_2020 | — | — | ρ=+0.2297 | β=— | — |
| ADRB2_HUMAN_Jones_2020 | — | — | ρ=+0.4971 | β=— | — |
| AICDA_HUMAN_Gajula_2014_3cycles | — | — | ρ=+0.2898 | β=— | — |
| AMFR_HUMAN_Tsuboyama_2023_4G3O | — | — | ρ=+0.2737 | β=— | — |
| AMIE_PSEAE_Wrenbeck_2017 | — | — | ρ=+0.5533 | β=— | — |
| ANCSZ_Hobbs_2022 | — | — | ρ=+0.5948 | β=— | — |
| ARGR_ECOLI_Tsuboyama_2023_1AOY | — | — | ρ=+0.5101 | β=— | — |
| B2L11_HUMAN_Dutta_2010_binding-Mcl-1 | — | — | ρ=+0.3215 | β=— | — |
| BBC1_YEAST_Tsuboyama_2023_1TG0 | — | — | ρ=+0.4737 | β=— | — |
| BCHB_CHLTE_Tsuboyama_2023_2KRU | — | — | ρ=+0.5409 | β=— | — |
| BLAT_ECOLX_Deng_2012 | — | — | ρ=+0.5291 | β=— | — |
| BLAT_ECOLX_Firnberg_2014 | — | — | ρ=+0.7279 | β=— | — |
| BLAT_ECOLX_Jacquier_2013 | — | — | ρ=+0.6933 | β=— | — |
| BLAT_ECOLX_Stiffler_2015 | — | — | ρ=+0.7227 | β=— | — |
| BRCA1_HUMAN_Findlay_2018 | — | — | ρ=+0.5137 | β=— | — |
| BRCA2_HUMAN_Erwood_2022_HEK293T | — | — | ρ=+0.4836 | β=— | — |
| C6KNH7_9INFA_Lee_2018 | — | — | ρ=+0.4636 | β=— | — |
| CALM1_HUMAN_Weile_2017 | — | — | ρ=+0.2100 | β=— | — |
| CAPSD_AAV2S_Sinai_2021 | — | — | ρ=+0.1999 | β=— | — |
| CAR11_HUMAN_Meitlis_2020_gof | — | — | ρ=+0.3284 | β=— | — |
| CAR11_HUMAN_Meitlis_2020_lof | — | — | ρ=+0.5003 | β=— | — |
| CAS9_STRP1_Spencer_2017_positive | — | — | ρ=+0.1839 | β=— | — |
| CASP3_HUMAN_Roychowdhury_2020 | — | — | ρ=+0.6326 | β=— | — |
| CASP7_HUMAN_Roychowdhury_2020 | — | — | ρ=+0.6139 | β=— | — |
| CATR_CHLRE_Tsuboyama_2023_2AMI | — | — | ρ=+0.6601 | β=— | — |
| CBPA2_HUMAN_Tsuboyama_2023_1O6X | — | — | ρ=+0.7622 | β=— | — |
| CBS_HUMAN_Sun_2020 | — | — | ρ=+0.3368 | β=— | — |
| CBX4_HUMAN_Tsuboyama_2023_2K28 | — | — | ρ=+0.6709 | β=— | — |
| CCDB_ECOLI_Adkar_2012 | — | — | ρ=+0.4783 | β=— | — |
| CCDB_ECOLI_Tripathi_2016 | — | — | ρ=+0.5141 | β=— | — |
| CCR5_HUMAN_Gill_2023 | — | — | ρ=+0.3448 | β=— | — |
| CD19_HUMAN_Klesmith_2019_FMC_singles | — | — | ρ=+0.0527 | β=— | — |
| CP2C9_HUMAN_Amorosi_2021_abundance | — | — | ρ=+0.6279 | β=— | — |
| CP2C9_HUMAN_Amorosi_2021_activity | — | — | ρ=+0.6720 | β=— | — |
| CSN4_MOUSE_Tsuboyama_2023_1UFM | — | — | ρ=+0.5214 | β=— | — |
| CUE1_YEAST_Tsuboyama_2023_2MYX | — | — | ρ=+0.5170 | β=— | — |
| D7PM05_CLYGR_Somermeyer_2022 | — | — | ρ=+0.4473 | β=— | — |
| DLG4_HUMAN_Faure_2021 | — | — | ρ=+0.5206 | β=— | — |
| DLG4_RAT_McLaughlin_2012 | — | — | ρ=+0.4391 | β=— | — |
| DN7A_SACS2_Tsuboyama_2023_1JIC | — | — | ρ=+0.3195 | β=— | — |
| DNJA1_HUMAN_Tsuboyama_2023_2LO1 | — | — | ρ=+0.7912 | β=— | — |
| DOCK1_MOUSE_Tsuboyama_2023_2M0Y | — | — | ρ=+0.5191 | β=— | — |
| DYR_ECOLI_Nguyen_2023 | — | — | ρ=+0.5384 | β=— | — |
| DYR_ECOLI_Thompson_2019 | — | — | ρ=+0.4793 | β=— | — |
| ENV_HV1B9_DuenasDecamp_2016 | — | — | ρ=+0.0422 | β=— | — |
| ENV_HV1BR_Haddox_2016 | — | — | ρ=+0.0359 | β=— | — |
| ENVZ_ECOLI_Ghose_2023 | — | — | ρ=+0.1733 | β=— | — |
| EPHB2_HUMAN_Tsuboyama_2023_1F0M | — | — | ρ=+0.8147 | β=— | — |
| ERBB2_HUMAN_Elazar_2016 | — | — | ρ=+0.4268 | β=— | — |
| ESTA_BACSU_Nutschel_2020 | — | — | ρ=+0.3002 | β=— | — |
| F7YBW8_MESOW_Ding_2023 | — | — | ρ=+0.4933 | β=— | — |
| F7YBW8_MESOW_Aakre_2015 | — | — | ρ=+0.4160 | β=— | — |
| FECA_ECOLI_Tsuboyama_2023_2D1U | — | — | ρ=+0.4970 | β=— | — |
| FKBP3_HUMAN_Tsuboyama_2023_2KFV | — | — | ρ=+0.2176 | β=— | — |
| GAL4_YEAST_Kitzman_2015 | — | — | ρ=+0.6558 | β=— | — |
| GCN4_YEAST_Staller_2018 | — | — | ρ=+0.2619 | β=— | — |
| GDIA_HUMAN_Silverstein_2021 | — | — | ρ=+0.3784 | β=— | — |
| GFP_AEQVI_Sarkisyan_2016 | — | — | ρ=+0.5256 | β=— | — |
| GLPA_HUMAN_Elazar_2016 | — | — | ρ=+0.4065 | β=— | — |
| GRB2_HUMAN_Faure_2021 | — | — | ρ=+0.6477 | β=— | — |
| HCP_LAMBD_Tsuboyama_2023_2L6Q | — | — | ρ=+0.6955 | β=— | — |
| HECD1_HUMAN_Tsuboyama_2023_3DKM | — | — | ρ=+0.3899 | β=— | — |
| HEM3_HUMAN_Loggerenberg_2023 | — | — | ρ=+0.3847 | β=— | — |
| HIS7_YEAST_Pokusaeva_2019 | — | — | ρ=+0.4197 | β=— | — |
| HMDH_HUMAN_Jiang_2019 | — | — | ρ=+0.5042 | β=— | — |
| HSP82_YEAST_Cote-Hammarlof_2020_growth-H2O2 | — | — | ρ=+0.2401 | β=— | — |
| HSP82_YEAST_Flynn_2019 | — | — | ρ=+0.2760 | β=— | — |
| HSP82_YEAST_Mishra_2016 | — | — | ρ=+0.4359 | β=— | — |
| HXK4_HUMAN_Gersing_2022_activity | — | — | ρ=+0.5121 | β=— | — |
| HXK4_HUMAN_Gersing_2023_abundance | — | — | ρ=+0.3720 | β=— | — |
| I6TAH8_I68A0_Doud_2015 | — | — | ρ=+0.0168 | β=— | — |
| IF1_ECOLI_Kelsic_2016 | — | — | ρ=+0.5794 | β=— | — |
| ILF3_HUMAN_Tsuboyama_2023_2L33 | — | — | ρ=+0.2802 | β=— | — |
| ISDH_STAAW_Tsuboyama_2023_2LHR | — | — | ρ=+0.4297 | β=— | — |
| KCNE1_HUMAN_Muhammad_2023_expression | — | — | ρ=+0.1017 | β=— | — |
| KCNE1_HUMAN_Muhammad_2023_function | — | — | ρ=+0.5273 | β=— | — |
| KCNH2_HUMAN_Kozek_2020 | — | — | ρ=+0.5202 | β=— | — |
| KCNJ2_MOUSE_Coyote-Maestas_2022_function | — | — | ρ=+0.3714 | β=— | — |
| KCNJ2_MOUSE_Coyote-Maestas_2022_surface | — | — | ρ=+0.3454 | β=— | — |
| KKA2_KLEPN_Melnikov_2014 | — | — | ρ=+0.5793 | β=— | — |
| LGK_LIPST_Klesmith_2015 | — | — | ρ=+0.5164 | β=— | — |
| LYAM1_HUMAN_Elazar_2016 | — | — | ρ=+0.3212 | β=— | — |
| MAFG_MOUSE_Tsuboyama_2023_1K1V | — | — | ρ=+0.6192 | β=— | — |
| MBD11_ARATH_Tsuboyama_2023_6ACV | — | — | ρ=+0.7400 | β=— | — |
| MET_HUMAN_Estevam_2023 | — | — | ρ=+0.5885 | β=— | — |
| MK01_HUMAN_Brenan_2016 | — | — | ρ=+0.1639 | β=— | — |
| MLAC_ECOLI_MacRae_2023 | — | — | ρ=+0.3660 | β=— | — |
| MSH2_HUMAN_Jia_2020 | — | — | ρ=+0.3187 | β=— | — |
| MTH3_HAEAE_RockahShmuel_2015 | — | — | ρ=+0.5018 | β=— | — |
| MTHR_HUMAN_Weile_2021 | — | — | ρ=+0.3222 | β=— | — |
| MYO3_YEAST_Tsuboyama_2023_2BTT | — | — | ρ=+0.5452 | β=— | — |
| NCAP_I34A1_Doud_2015 | — | — | ρ=+0.0200 | β=— | — |
| NKX31_HUMAN_Tsuboyama_2023_2L9R | — | — | ρ=+0.6714 | β=— | — |
| NPC1_HUMAN_Erwood_2022_HEK293T | — | — | ρ=+0.7011 | β=— | — |
| NPC1_HUMAN_Erwood_2022_RPE1 | — | — | ρ=+0.6933 | β=— | — |
| NRAM_I33A0_Jiang_2016 | — | — | ρ=+0.1663 | β=— | — |
| NUD15_HUMAN_Suiter_2020 | — | — | ρ=+0.5281 | β=— | — |
| NUSA_ECOLI_Tsuboyama_2023_1WCL | — | — | ρ=+0.6317 | β=— | — |
| NUSG_MYCTU_Tsuboyama_2023_2MI6 | — | — | ρ=+0.4837 | β=— | — |
| OBSCN_HUMAN_Tsuboyama_2023_1V1C | — | — | ρ=+0.7795 | β=— | — |
| ODP2_GEOSE_Tsuboyama_2023_1W4G | — | — | ρ=+0.0544 | β=— | — |
| OPSD_HUMAN_Wan_2019 | — | — | ρ=+0.4766 | β=— | — |
| OTC_HUMAN_Lo_2023 | — | — | ρ=+0.5229 | β=— | — |
| OTU7A_HUMAN_Tsuboyama_2023_2L2D | — | — | ρ=+0.3338 | β=— | — |
| OXDA_RHOTO_Vanella_2023_activity | — | — | ρ=+0.3544 | β=— | — |
| OXDA_RHOTO_Vanella_2023_expression | — | — | ρ=+0.2987 | β=— | — |
| P53_HUMAN_Giacomelli_2018_Null_Etoposide | — | — | ρ=+0.4243 | β=— | — |
| P53_HUMAN_Giacomelli_2018_Null_Nutlin | — | — | ρ=+0.3969 | β=— | — |
| P53_HUMAN_Giacomelli_2018_WT_Nutlin | — | — | ρ=+0.3843 | β=— | — |
| P53_HUMAN_Kotler_2018 | — | — | ρ=+0.6624 | β=— | — |
| P84126_THETH_Chan_2017 | — | — | ρ=+0.5869 | β=— | — |
| PA_I34A1_Wu_2015 | — | — | ρ=+0.0385 | β=— | — |
| PABP_YEAST_Melamed_2013 | — | — | ρ=+0.6821 | β=— | — |
| PAI1_HUMAN_Huttinger_2021 | — | — | ρ=+0.4444 | β=— | — |
| PHOT_CHLRE_Chen_2023 | — | — | ρ=+0.5995 | β=— | — |
| PIN1_HUMAN_Tsuboyama_2023_1I6C | — | — | ρ=+0.6865 | β=— | — |
| PITX2_HUMAN_Tsuboyama_2023_2L7M | — | — | ρ=+0.6024 | β=— | — |
| PKN1_HUMAN_Tsuboyama_2023_1URF | — | — | ρ=+0.3053 | β=— | — |
| POLG_CXB3N_Mattenberger_2021 | — | — | ρ=+0.3814 | β=— | — |
| POLG_DEN26_Suphatrakul_2023 | — | — | ρ=+0.1426 | β=— | — |
| POLG_HCVJF_Qi_2014 | — | — | ρ=+0.1336 | β=— | — |
| POLG_PESV_Tsuboyama_2023_2MXD | — | — | ρ=+0.4059 | β=— | — |
| PPARG_HUMAN_Majithia_2016 | — | — | ρ=+0.5900 | β=— | — |
| PPM1D_HUMAN_Miller_2022 | — | — | ρ=+0.5933 | β=— | — |
| PR40A_HUMAN_Tsuboyama_2023_1UZC | — | — | ρ=+0.7994 | β=— | — |
| PRKN_HUMAN_Clausen_2023 | — | — | ρ=+0.5029 | β=— | — |
| PSAE_PICP2_Tsuboyama_2023_1PSE | — | — | ρ=+0.7020 | β=— | — |
| PTEN_HUMAN_Matreyek_2021 | — | — | ρ=+0.4644 | β=— | — |
| PTEN_HUMAN_Mighell_2018 | — | — | ρ=+0.4927 | β=— | — |
| Q2N0S5_9HIV1_Haddox_2018 | — | — | ρ=+0.0283 | β=— | — |
| Q53Z42_HUMAN_McShan_2019_binding-TAPBPR | — | — | ρ=+0.3219 | β=— | — |
| Q53Z42_HUMAN_McShan_2019_expression | — | — | ρ=+0.5596 | β=— | — |
| Q59976_STRSQ_Romero_2015 | — | — | ρ=+0.5546 | β=— | — |
| Q6WV12_9MAXI_Somermeyer_2022 | — | — | ρ=+0.2384 | β=— | — |
| Q837P4_ENTFA_Meier_2023 | — | — | ρ=+0.5109 | β=— | — |
| Q837P5_ENTFA_Meier_2023 | — | — | ρ=+0.3900 | β=— | — |
| Q8WTC7_9CNID_Somermeyer_2022 | — | — | ρ=+0.2231 | β=— | — |
| R1AB_SARS2_Flynn_2022 | — | — | ρ=+0.1180 | β=— | — |
| RAD_ANTMA_Tsuboyama_2023_2CJJ | — | — | ρ=+0.5176 | β=— | — |
| RAF1_HUMAN_Zinkus-Boltz_2019 | — | — | ρ=+0.4490 | β=— | — |
| RASH_HUMAN_Bandaru_2017 | — | — | ρ=+0.4769 | β=— | — |
| RASK_HUMAN_Weng_2022_abundance | — | — | ρ=+0.3113 | β=— | — |
| RASK_HUMAN_Weng_2022_binding-DARPin_K55 | — | — | ρ=+0.6177 | β=— | — |
| RBP1_HUMAN_Tsuboyama_2023_2KWH | — | — | ρ=+0.5133 | β=— | — |
| RCD1_ARATH_Tsuboyama_2023_5OAO | — | — | ρ=+0.5487 | β=— | — |
| RCRO_LAMBD_Tsuboyama_2023_1ORC | — | — | ρ=+0.5658 | β=— | — |
| RD23A_HUMAN_Tsuboyama_2023_1IFY | — | — | ρ=+0.4695 | β=— | — |
| RDRP_I33A0_Li_2023 | — | — | ρ=+0.2904 | β=— | — |
| REV_HV1H2_Fernandes_2016 | — | — | ρ=+0.2265 | β=— | — |
| RFAH_ECOLI_Tsuboyama_2023_2LCL | — | — | ρ=+0.2959 | β=— | — |
| RL20_AQUAE_Tsuboyama_2023_1GYZ | — | — | ρ=+0.7216 | β=— | — |
| RL40A_YEAST_Mavor_2016 | — | — | ρ=+0.5246 | β=— | — |
| RL40A_YEAST_Roscoe_2013 | — | — | ρ=+0.5956 | β=— | — |
| RL40A_YEAST_Roscoe_2014 | — | — | ρ=+0.5280 | β=— | — |
| RNC_ECOLI_Weeks_2023 | — | — | ρ=+0.5835 | β=— | — |
| RPC1_BP434_Tsuboyama_2023_1R69 | — | — | ρ=+0.7046 | β=— | — |
| RPC1_LAMBD_Li_2019_high-expression | — | — | ρ=+0.5346 | β=— | — |
| RPC1_LAMBD_Li_2019_low-expression | — | — | ρ=+0.4947 | β=— | — |
| RS15_GEOSE_Tsuboyama_2023_1A32 | — | — | ρ=+0.4166 | β=— | — |
| S22A1_HUMAN_Yee_2023_abundance | — | — | ρ=+0.6183 | β=— | — |
| S22A1_HUMAN_Yee_2023_activity | — | — | ρ=+0.5597 | β=— | — |
| SAV1_MOUSE_Tsuboyama_2023_2YSB | — | — | ρ=+0.4742 | β=— | — |
| SBI_STAAM_Tsuboyama_2023_2JVG | — | — | ρ=+0.5221 | β=— | — |
| SC6A4_HUMAN_Young_2021 | — | — | ρ=+0.5420 | β=— | — |
| SCIN_STAAR_Tsuboyama_2023_2QFF | — | — | ρ=+0.3774 | β=— | — |
| SCN5A_HUMAN_Glazer_2019 | — | — | ρ=+0.1214 | β=— | — |
| SDA_BACSU_Tsuboyama_2023_1PV0 | — | — | ρ=+0.6044 | β=— | — |
| SERC_HUMAN_Xie_2023 | — | — | ρ=+0.5483 | β=— | — |
| SHOC2_HUMAN_Kwon_2022 | — | — | ρ=+0.4073 | β=— | — |
| SOX30_HUMAN_Tsuboyama_2023_7JJK | — | — | ρ=+0.3327 | β=— | — |
| SPA_STAAU_Tsuboyama_2023_1LP1 | — | — | ρ=+0.3949 | β=— | — |
| SPG1_STRSG_Olson_2014 | — | — | ρ=+0.3186 | β=— | — |
| SPG1_STRSG_Wu_2016 | — | — | ρ=+0.1374 | β=— | — |
| SPG2_STRSG_Tsuboyama_2023_5UBS | — | — | ρ=+0.5450 | β=— | — |
| SPIKE_SARS2_Starr_2020_binding | — | — | ρ=-0.0200 | β=— | — |
| SPIKE_SARS2_Starr_2020_expression | — | — | ρ=+0.0184 | β=— | — |
| SPTN1_CHICK_Tsuboyama_2023_1TUD | — | — | ρ=+0.6734 | β=— | — |
| SQSTM_MOUSE_Tsuboyama_2023_2RRU | — | — | ρ=+0.6376 | β=— | — |
| SR43C_ARATH_Tsuboyama_2023_2N88 | — | — | ρ=+0.6758 | β=— | — |
| SRBS1_HUMAN_Tsuboyama_2023_2O2W | — | — | ρ=+0.7447 | β=— | — |
| SRC_HUMAN_Ahler_2019 | — | — | ρ=+0.5472 | β=— | — |
| SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM | — | — | ρ=+0.4557 | β=— | — |
| SRC_HUMAN_Nguyen_2022 | — | — | ρ=+0.4587 | β=— | — |
| SUMO1_HUMAN_Weile_2017 | — | — | ρ=+0.5008 | β=— | — |
| SYUA_HUMAN_Newberry_2020 | — | — | ρ=+0.1280 | β=— | — |
| TADBP_HUMAN_Bolognesi_2019 | — | — | ρ=-0.1187 | β=— | — |
| TAT_HV1BR_Fernandes_2016 | — | — | ρ=-0.0447 | β=— | — |
| TCRG1_MOUSE_Tsuboyama_2023_1E0L | — | — | ρ=+0.7552 | β=— | — |
| THO1_YEAST_Tsuboyama_2023_2WQG | — | — | ρ=+0.5267 | β=— | — |
| TNKS2_HUMAN_Tsuboyama_2023_5JRT | — | — | ρ=+0.5228 | β=— | — |
| TPK1_HUMAN_Weile_2017 | — | — | ρ=+0.3334 | β=— | — |
| TPMT_HUMAN_Matreyek_2018 | — | — | ρ=+0.5371 | β=— | — |
| TPOR_HUMAN_Bridgford_2020 | — | — | ρ=+0.2949 | β=— | — |
| TRPC_SACS2_Chan_2017 | — | — | ρ=+0.6441 | β=— | — |
| TRPC_THEMA_Chan_2017 | — | — | ρ=+0.4592 | β=— | — |
| UBC9_HUMAN_Weile_2017 | — | — | ρ=+0.4546 | β=— | — |
| UBE4B_HUMAN_Tsuboyama_2023_3L1X | — | — | ρ=+0.4489 | β=— | — |
| UBE4B_MOUSE_Starita_2013 | — | — | ρ=+0.4138 | β=— | — |
| UBR5_HUMAN_Tsuboyama_2023_1I2T | — | — | ρ=+0.5532 | β=— | — |
| VG08_BPP22_Tsuboyama_2023_2GP8 | — | — | ρ=+0.6618 | β=— | — |
| VILI_CHICK_Tsuboyama_2023_1YU5 | — | — | ρ=+0.6605 | β=— | — |
| VKOR1_HUMAN_Chiasson_2020_abundance | — | — | ρ=+0.5033 | β=— | — |
| VKOR1_HUMAN_Chiasson_2020_activity | — | — | ρ=+0.4131 | β=— | — |
| VRPI_BPT7_Tsuboyama_2023_2WNM | — | — | ρ=+0.5756 | β=— | — |
| YAIA_ECOLI_Tsuboyama_2023_2KVT | — | — | ρ=+0.6149 | β=— | — |
| YAP1_HUMAN_Araya_2012 | — | — | ρ=+0.4227 | β=— | — |
| YNZC_BACSU_Tsuboyama_2023_2JVD | — | — | ρ=+0.6778 | β=— | — |


---

###  · run · `masked_marginals` on NVIDIA A100-SXM4-40GB

<!-- [run:masked_marginals:A0A140D2T1_ZIKV_Sourisseau_2019,A0A192B1T2_9HIV1_Haddox_2018,A0A1I9GEU1_NEIME_Kennouche_2019,A0A247D711_LISMN_Stadelmann_2021,A0A2Z5U3Z0_9INFA_Doud_2016,A0A2Z5U3Z0_9INFA_Wu_2014,A4D664_9INFA_Soh_2019,A4GRB6_PSEAI_Chen_2020,A4_HUMAN_Seuma_2022,AACC1_PSEAI_Dandage_2018,ACE2_HUMAN_Chan_2020,ADRB2_HUMAN_Jones_2020,AICDA_HUMAN_Gajula_2014_3cycles,AMFR_HUMAN_Tsuboyama_2023_4G3O,AMIE_PSEAE_Wrenbeck_2017,ANCSZ_Hobbs_2022,ARGR_ECOLI_Tsuboyama_2023_1AOY,B2L11_HUMAN_Dutta_2010_binding-Mcl-1,BBC1_YEAST_Tsuboyama_2023_1TG0,BCHB_CHLTE_Tsuboyama_2023_2KRU,BLAT_ECOLX_Deng_2012,BLAT_ECOLX_Firnberg_2014,BLAT_ECOLX_Jacquier_2013,BLAT_ECOLX_Stiffler_2015,BRCA1_HUMAN_Findlay_2018,BRCA2_HUMAN_Erwood_2022_HEK293T,C6KNH7_9INFA_Lee_2018,CALM1_HUMAN_Weile_2017,CAPSD_AAV2S_Sinai_2021,CAR11_HUMAN_Meitlis_2020_gof,CAR11_HUMAN_Meitlis_2020_lof,CAS9_STRP1_Spencer_2017_positive,CASP3_HUMAN_Roychowdhury_2020,CASP7_HUMAN_Roychowdhury_2020,CATR_CHLRE_Tsuboyama_2023_2AMI,CBPA2_HUMAN_Tsuboyama_2023_1O6X,CBS_HUMAN_Sun_2020,CBX4_HUMAN_Tsuboyama_2023_2K28,CCDB_ECOLI_Adkar_2012,CCDB_ECOLI_Tripathi_2016,CCR5_HUMAN_Gill_2023,CD19_HUMAN_Klesmith_2019_FMC_singles,CP2C9_HUMAN_Amorosi_2021_abundance,CP2C9_HUMAN_Amorosi_2021_activity,CSN4_MOUSE_Tsuboyama_2023_1UFM,CUE1_YEAST_Tsuboyama_2023_2MYX,D7PM05_CLYGR_Somermeyer_2022,DLG4_HUMAN_Faure_2021,DLG4_RAT_McLaughlin_2012,DN7A_SACS2_Tsuboyama_2023_1JIC,DNJA1_HUMAN_Tsuboyama_2023_2LO1,DOCK1_MOUSE_Tsuboyama_2023_2M0Y,DYR_ECOLI_Nguyen_2023,DYR_ECOLI_Thompson_2019,ENVZ_ECOLI_Ghose_2023,ENV_HV1B9_DuenasDecamp_2016,ENV_HV1BR_Haddox_2016,EPHB2_HUMAN_Tsuboyama_2023_1F0M,ERBB2_HUMAN_Elazar_2016,ESTA_BACSU_Nutschel_2020,F7YBW8_MESOW_Aakre_2015,F7YBW8_MESOW_Ding_2023,FECA_ECOLI_Tsuboyama_2023_2D1U,FKBP3_HUMAN_Tsuboyama_2023_2KFV,GAL4_YEAST_Kitzman_2015,GCN4_YEAST_Staller_2018,GDIA_HUMAN_Silverstein_2021,GFP_AEQVI_Sarkisyan_2016,GLPA_HUMAN_Elazar_2016,GRB2_HUMAN_Faure_2021,HCP_LAMBD_Tsuboyama_2023_2L6Q,HECD1_HUMAN_Tsuboyama_2023_3DKM,HEM3_HUMAN_Loggerenberg_2023,HIS7_YEAST_Pokusaeva_2019,HMDH_HUMAN_Jiang_2019,HSP82_YEAST_Cote-Hammarlof_2020_growth-H2O2,HSP82_YEAST_Flynn_2019,HSP82_YEAST_Mishra_2016,HXK4_HUMAN_Gersing_2022_activity,HXK4_HUMAN_Gersing_2023_abundance,I6TAH8_I68A0_Doud_2015,IF1_ECOLI_Kelsic_2016,ILF3_HUMAN_Tsuboyama_2023_2L33,ISDH_STAAW_Tsuboyama_2023_2LHR,KCNE1_HUMAN_Muhammad_2023_expression,KCNE1_HUMAN_Muhammad_2023_function,KCNH2_HUMAN_Kozek_2020,KCNJ2_MOUSE_Coyote-Maestas_2022_function,KCNJ2_MOUSE_Coyote-Maestas_2022_surface,KKA2_KLEPN_Melnikov_2014,LGK_LIPST_Klesmith_2015,LYAM1_HUMAN_Elazar_2016,MAFG_MOUSE_Tsuboyama_2023_1K1V,MBD11_ARATH_Tsuboyama_2023_6ACV,MET_HUMAN_Estevam_2023,MK01_HUMAN_Brenan_2016,MLAC_ECOLI_MacRae_2023,MSH2_HUMAN_Jia_2020,MTH3_HAEAE_RockahShmuel_2015,MTHR_HUMAN_Weile_2021,MYO3_YEAST_Tsuboyama_2023_2BTT,NCAP_I34A1_Doud_2015,NKX31_HUMAN_Tsuboyama_2023_2L9R,NPC1_HUMAN_Erwood_2022_HEK293T,NPC1_HUMAN_Erwood_2022_RPE1,NRAM_I33A0_Jiang_2016,NUD15_HUMAN_Suiter_2020,NUSA_ECOLI_Tsuboyama_2023_1WCL,NUSG_MYCTU_Tsuboyama_2023_2MI6,OBSCN_HUMAN_Tsuboyama_2023_1V1C,ODP2_GEOSE_Tsuboyama_2023_1W4G,OPSD_HUMAN_Wan_2019,OTC_HUMAN_Lo_2023,OTU7A_HUMAN_Tsuboyama_2023_2L2D,OXDA_RHOTO_Vanella_2023_activity,OXDA_RHOTO_Vanella_2023_expression,P53_HUMAN_Giacomelli_2018_Null_Etoposide,P53_HUMAN_Giacomelli_2018_Null_Nutlin,P53_HUMAN_Giacomelli_2018_WT_Nutlin,P53_HUMAN_Kotler_2018,P84126_THETH_Chan_2017,PABP_YEAST_Melamed_2013,PAI1_HUMAN_Huttinger_2021,PA_I34A1_Wu_2015,PHOT_CHLRE_Chen_2023,PIN1_HUMAN_Tsuboyama_2023_1I6C,PITX2_HUMAN_Tsuboyama_2023_2L7M,PKN1_HUMAN_Tsuboyama_2023_1URF,POLG_CXB3N_Mattenberger_2021,POLG_DEN26_Suphatrakul_2023,POLG_HCVJF_Qi_2014,POLG_PESV_Tsuboyama_2023_2MXD,PPARG_HUMAN_Majithia_2016,PPM1D_HUMAN_Miller_2022,PR40A_HUMAN_Tsuboyama_2023_1UZC,PRKN_HUMAN_Clausen_2023,PSAE_PICP2_Tsuboyama_2023_1PSE,PTEN_HUMAN_Matreyek_2021,PTEN_HUMAN_Mighell_2018,Q2N0S5_9HIV1_Haddox_2018,Q53Z42_HUMAN_McShan_2019_binding-TAPBPR,Q53Z42_HUMAN_McShan_2019_expression,Q59976_STRSQ_Romero_2015,Q6WV12_9MAXI_Somermeyer_2022,Q837P4_ENTFA_Meier_2023,Q837P5_ENTFA_Meier_2023,Q8WTC7_9CNID_Somermeyer_2022,R1AB_SARS2_Flynn_2022,RAD_ANTMA_Tsuboyama_2023_2CJJ,RAF1_HUMAN_Zinkus-Boltz_2019,RASH_HUMAN_Bandaru_2017,RASK_HUMAN_Weng_2022_abundance,RASK_HUMAN_Weng_2022_binding-DARPin_K55,RBP1_HUMAN_Tsuboyama_2023_2KWH,RCD1_ARATH_Tsuboyama_2023_5OAO,RCRO_LAMBD_Tsuboyama_2023_1ORC,RD23A_HUMAN_Tsuboyama_2023_1IFY,RDRP_I33A0_Li_2023,REV_HV1H2_Fernandes_2016,RFAH_ECOLI_Tsuboyama_2023_2LCL,RL20_AQUAE_Tsuboyama_2023_1GYZ,RL40A_YEAST_Mavor_2016,RL40A_YEAST_Roscoe_2013,RL40A_YEAST_Roscoe_2014,RNC_ECOLI_Weeks_2023,RPC1_BP434_Tsuboyama_2023_1R69,RPC1_LAMBD_Li_2019_high-expression,RPC1_LAMBD_Li_2019_low-expression,RS15_GEOSE_Tsuboyama_2023_1A32,S22A1_HUMAN_Yee_2023_abundance,S22A1_HUMAN_Yee_2023_activity,SAV1_MOUSE_Tsuboyama_2023_2YSB,SBI_STAAM_Tsuboyama_2023_2JVG,SC6A4_HUMAN_Young_2021,SCIN_STAAR_Tsuboyama_2023_2QFF,SCN5A_HUMAN_Glazer_2019,SDA_BACSU_Tsuboyama_2023_1PV0,SERC_HUMAN_Xie_2023,SHOC2_HUMAN_Kwon_2022,SOX30_HUMAN_Tsuboyama_2023_7JJK,SPA_STAAU_Tsuboyama_2023_1LP1,SPG1_STRSG_Olson_2014,SPG1_STRSG_Wu_2016,SPG2_STRSG_Tsuboyama_2023_5UBS,SPIKE_SARS2_Starr_2020_binding,SPIKE_SARS2_Starr_2020_expression,SPTN1_CHICK_Tsuboyama_2023_1TUD,SQSTM_MOUSE_Tsuboyama_2023_2RRU,SR43C_ARATH_Tsuboyama_2023_2N88,SRBS1_HUMAN_Tsuboyama_2023_2O2W,SRC_HUMAN_Ahler_2019,SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM,SRC_HUMAN_Nguyen_2022,SUMO1_HUMAN_Weile_2017,SYUA_HUMAN_Newberry_2020,TADBP_HUMAN_Bolognesi_2019,TAT_HV1BR_Fernandes_2016,TCRG1_MOUSE_Tsuboyama_2023_1E0L,THO1_YEAST_Tsuboyama_2023_2WQG,TNKS2_HUMAN_Tsuboyama_2023_5JRT,TPK1_HUMAN_Weile_2017,TPMT_HUMAN_Matreyek_2018,TPOR_HUMAN_Bridgford_2020,TRPC_SACS2_Chan_2017,TRPC_THEMA_Chan_2017,UBC9_HUMAN_Weile_2017,UBE4B_HUMAN_Tsuboyama_2023_3L1X,UBE4B_MOUSE_Starita_2013,UBR5_HUMAN_Tsuboyama_2023_1I2T,VG08_BPP22_Tsuboyama_2023_2GP8,VILI_CHICK_Tsuboyama_2023_1YU5,VKOR1_HUMAN_Chiasson_2020_abundance,VKOR1_HUMAN_Chiasson_2020_activity,VRPI_BPT7_Tsuboyama_2023_2WNM,YAIA_ECOLI_Tsuboyama_2023_2KVT,YAP1_HUMAN_Araya_2012,YNZC_BACSU_Tsuboyama_2023_2JVD] -->

**Method run: `masked_marginals` — 217 assays — mean ρ = +0.4391**

Wall total: 0.0s · Hardware: NVIDIA A100-SXM4-40GB · Integrity: Merkle `889c170fe685c443` · fair-esm@`2b369911`

| Assay | L | N | ρ | β (s/AA²) | Wall |
|---|---|---|---|---|---|
| A0A140D2T1_ZIKV_Sourisseau_2019 | — | — | ρ=+0.2159 | β=— | — |
| A0A192B1T2_9HIV1_Haddox_2018 | — | — | ρ=+0.0798 | β=— | — |
| A0A1I9GEU1_NEIME_Kennouche_2019 | — | — | ρ=+0.0297 | β=— | — |
| A0A247D711_LISMN_Stadelmann_2021 | — | — | ρ=+0.0661 | β=— | — |
| A0A2Z5U3Z0_9INFA_Doud_2016 | — | — | ρ=+0.5067 | β=— | — |
| A0A2Z5U3Z0_9INFA_Wu_2014 | — | — | ρ=+0.4642 | β=— | — |
| A4_HUMAN_Seuma_2022 | — | — | ρ=+0.4311 | β=— | — |
| A4D664_9INFA_Soh_2019 | — | — | ρ=+0.1488 | β=— | — |
| A4GRB6_PSEAI_Chen_2020 | — | — | ρ=+0.7382 | β=— | — |
| AACC1_PSEAI_Dandage_2018 | — | — | ρ=+0.4904 | β=— | — |
| ACE2_HUMAN_Chan_2020 | — | — | ρ=+0.2245 | β=— | — |
| ADRB2_HUMAN_Jones_2020 | — | — | ρ=+0.4926 | β=— | — |
| AICDA_HUMAN_Gajula_2014_3cycles | — | — | ρ=+0.3284 | β=— | — |
| AMFR_HUMAN_Tsuboyama_2023_4G3O | — | — | ρ=+0.2610 | β=— | — |
| AMIE_PSEAE_Wrenbeck_2017 | — | — | ρ=+0.5569 | β=— | — |
| ANCSZ_Hobbs_2022 | — | — | ρ=+0.6093 | β=— | — |
| ARGR_ECOLI_Tsuboyama_2023_1AOY | — | — | ρ=+0.4965 | β=— | — |
| B2L11_HUMAN_Dutta_2010_binding-Mcl-1 | — | — | ρ=+0.2696 | β=— | — |
| BBC1_YEAST_Tsuboyama_2023_1TG0 | — | — | ρ=+0.4792 | β=— | — |
| BCHB_CHLTE_Tsuboyama_2023_2KRU | — | — | ρ=+0.5097 | β=— | — |
| BLAT_ECOLX_Deng_2012 | — | — | ρ=+0.5281 | β=— | — |
| BLAT_ECOLX_Firnberg_2014 | — | — | ρ=+0.7371 | β=— | — |
| BLAT_ECOLX_Jacquier_2013 | — | — | ρ=+0.7038 | β=— | — |
| BLAT_ECOLX_Stiffler_2015 | — | — | ρ=+0.7315 | β=— | — |
| BRCA1_HUMAN_Findlay_2018 | — | — | ρ=+0.5153 | β=— | — |
| BRCA2_HUMAN_Erwood_2022_HEK293T | — | — | ρ=+0.5106 | β=— | — |
| C6KNH7_9INFA_Lee_2018 | — | — | ρ=+0.4830 | β=— | — |
| CALM1_HUMAN_Weile_2017 | — | — | ρ=+0.2116 | β=— | — |
| CAPSD_AAV2S_Sinai_2021 | — | — | ρ=+0.2767 | β=— | — |
| CAR11_HUMAN_Meitlis_2020_gof | — | — | ρ=+0.3272 | β=— | — |
| CAR11_HUMAN_Meitlis_2020_lof | — | — | ρ=+0.5015 | β=— | — |
| CAS9_STRP1_Spencer_2017_positive | — | — | ρ=+0.1821 | β=— | — |
| CASP3_HUMAN_Roychowdhury_2020 | — | — | ρ=+0.6375 | β=— | — |
| CASP7_HUMAN_Roychowdhury_2020 | — | — | ρ=+0.6221 | β=— | — |
| CATR_CHLRE_Tsuboyama_2023_2AMI | — | — | ρ=+0.6582 | β=— | — |
| CBPA2_HUMAN_Tsuboyama_2023_1O6X | — | — | ρ=+0.7017 | β=— | — |
| CBS_HUMAN_Sun_2020 | — | — | ρ=+0.3422 | β=— | — |
| CBX4_HUMAN_Tsuboyama_2023_2K28 | — | — | ρ=+0.6815 | β=— | — |
| CCDB_ECOLI_Adkar_2012 | — | — | ρ=+0.4659 | β=— | — |
| CCDB_ECOLI_Tripathi_2016 | — | — | ρ=+0.5115 | β=— | — |
| CCR5_HUMAN_Gill_2023 | — | — | ρ=+0.3474 | β=— | — |
| CD19_HUMAN_Klesmith_2019_FMC_singles | — | — | ρ=+0.0499 | β=— | — |
| CP2C9_HUMAN_Amorosi_2021_abundance | — | — | ρ=+0.6345 | β=— | — |
| CP2C9_HUMAN_Amorosi_2021_activity | — | — | ρ=+0.6789 | β=— | — |
| CSN4_MOUSE_Tsuboyama_2023_1UFM | — | — | ρ=+0.4838 | β=— | — |
| CUE1_YEAST_Tsuboyama_2023_2MYX | — | — | ρ=+0.5467 | β=— | — |
| D7PM05_CLYGR_Somermeyer_2022 | — | — | ρ=+0.0500 | β=— | — |
| DLG4_HUMAN_Faure_2021 | — | — | ρ=+0.5844 | β=— | — |
| DLG4_RAT_McLaughlin_2012 | — | — | ρ=+0.5428 | β=— | — |
| DN7A_SACS2_Tsuboyama_2023_1JIC | — | — | ρ=+0.3371 | β=— | — |
| DNJA1_HUMAN_Tsuboyama_2023_2LO1 | — | — | ρ=+0.8029 | β=— | — |
| DOCK1_MOUSE_Tsuboyama_2023_2M0Y | — | — | ρ=+0.5315 | β=— | — |
| DYR_ECOLI_Nguyen_2023 | — | — | ρ=+0.5442 | β=— | — |
| DYR_ECOLI_Thompson_2019 | — | — | ρ=+0.4800 | β=— | — |
| ENV_HV1B9_DuenasDecamp_2016 | — | — | ρ=+0.0122 | β=— | — |
| ENV_HV1BR_Haddox_2016 | — | — | ρ=+0.0458 | β=— | — |
| ENVZ_ECOLI_Ghose_2023 | — | — | ρ=+0.1716 | β=— | — |
| EPHB2_HUMAN_Tsuboyama_2023_1F0M | — | — | ρ=+0.8105 | β=— | — |
| ERBB2_HUMAN_Elazar_2016 | — | — | ρ=+0.4220 | β=— | — |
| ESTA_BACSU_Nutschel_2020 | — | — | ρ=+0.2995 | β=— | — |
| F7YBW8_MESOW_Ding_2023 | — | — | ρ=+0.4461 | β=— | — |
| F7YBW8_MESOW_Aakre_2015 | — | — | ρ=+0.3829 | β=— | — |
| FECA_ECOLI_Tsuboyama_2023_2D1U | — | — | ρ=+0.5237 | β=— | — |
| FKBP3_HUMAN_Tsuboyama_2023_2KFV | — | — | ρ=+0.1878 | β=— | — |
| GAL4_YEAST_Kitzman_2015 | — | — | ρ=+0.6676 | β=— | — |
| GCN4_YEAST_Staller_2018 | — | — | ρ=+0.2811 | β=— | — |
| GDIA_HUMAN_Silverstein_2021 | — | — | ρ=+0.3971 | β=— | — |
| GFP_AEQVI_Sarkisyan_2016 | — | — | ρ=+0.1078 | β=— | — |
| GLPA_HUMAN_Elazar_2016 | — | — | ρ=+0.4253 | β=— | — |
| GRB2_HUMAN_Faure_2021 | — | — | ρ=+0.6474 | β=— | — |
| HCP_LAMBD_Tsuboyama_2023_2L6Q | — | — | ρ=+0.6859 | β=— | — |
| HECD1_HUMAN_Tsuboyama_2023_3DKM | — | — | ρ=+0.3867 | β=— | — |
| HEM3_HUMAN_Loggerenberg_2023 | — | — | ρ=+0.3932 | β=— | — |
| HIS7_YEAST_Pokusaeva_2019 | — | — | ρ=+0.4113 | β=— | — |
| HMDH_HUMAN_Jiang_2019 | — | — | ρ=+0.5089 | β=— | — |
| HSP82_YEAST_Cote-Hammarlof_2020_growth-H2O2 | — | — | ρ=+0.2565 | β=— | — |
| HSP82_YEAST_Flynn_2019 | — | — | ρ=+0.2833 | β=— | — |
| HSP82_YEAST_Mishra_2016 | — | — | ρ=+0.4612 | β=— | — |
| HXK4_HUMAN_Gersing_2022_activity | — | — | ρ=+0.5166 | β=— | — |
| HXK4_HUMAN_Gersing_2023_abundance | — | — | ρ=+0.3674 | β=— | — |
| I6TAH8_I68A0_Doud_2015 | — | — | ρ=+0.0204 | β=— | — |
| IF1_ECOLI_Kelsic_2016 | — | — | ρ=+0.5987 | β=— | — |
| ILF3_HUMAN_Tsuboyama_2023_2L33 | — | — | ρ=+0.3179 | β=— | — |
| ISDH_STAAW_Tsuboyama_2023_2LHR | — | — | ρ=+0.4291 | β=— | — |
| KCNE1_HUMAN_Muhammad_2023_expression | — | — | ρ=+0.1060 | β=— | — |
| KCNE1_HUMAN_Muhammad_2023_function | — | — | ρ=+0.5247 | β=— | — |
| KCNH2_HUMAN_Kozek_2020 | — | — | ρ=+0.5112 | β=— | — |
| KCNJ2_MOUSE_Coyote-Maestas_2022_function | — | — | ρ=+0.3820 | β=— | — |
| KCNJ2_MOUSE_Coyote-Maestas_2022_surface | — | — | ρ=+0.3516 | β=— | — |
| KKA2_KLEPN_Melnikov_2014 | — | — | ρ=+0.6014 | β=— | — |
| LGK_LIPST_Klesmith_2015 | — | — | ρ=+0.5151 | β=— | — |
| LYAM1_HUMAN_Elazar_2016 | — | — | ρ=+0.3102 | β=— | — |
| MAFG_MOUSE_Tsuboyama_2023_1K1V | — | — | ρ=+0.4803 | β=— | — |
| MBD11_ARATH_Tsuboyama_2023_6ACV | — | — | ρ=+0.6987 | β=— | — |
| MET_HUMAN_Estevam_2023 | — | — | ρ=+0.5899 | β=— | — |
| MK01_HUMAN_Brenan_2016 | — | — | ρ=+0.1782 | β=— | — |
| MLAC_ECOLI_MacRae_2023 | — | — | ρ=+0.3821 | β=— | — |
| MSH2_HUMAN_Jia_2020 | — | — | ρ=+0.3395 | β=— | — |
| MTH3_HAEAE_RockahShmuel_2015 | — | — | ρ=+0.5270 | β=— | — |
| MTHR_HUMAN_Weile_2021 | — | — | ρ=+0.3475 | β=— | — |
| MYO3_YEAST_Tsuboyama_2023_2BTT | — | — | ρ=+0.5880 | β=— | — |
| NCAP_I34A1_Doud_2015 | — | — | ρ=+0.0278 | β=— | — |
| NKX31_HUMAN_Tsuboyama_2023_2L9R | — | — | ρ=+0.6511 | β=— | — |
| NPC1_HUMAN_Erwood_2022_HEK293T | — | — | ρ=+0.6946 | β=— | — |
| NPC1_HUMAN_Erwood_2022_RPE1 | — | — | ρ=+0.6604 | β=— | — |
| NRAM_I33A0_Jiang_2016 | — | — | ρ=+0.1608 | β=— | — |
| NUD15_HUMAN_Suiter_2020 | — | — | ρ=+0.5256 | β=— | — |
| NUSA_ECOLI_Tsuboyama_2023_1WCL | — | — | ρ=+0.4943 | β=— | — |
| NUSG_MYCTU_Tsuboyama_2023_2MI6 | — | — | ρ=+0.5184 | β=— | — |
| OBSCN_HUMAN_Tsuboyama_2023_1V1C | — | — | ρ=+0.7971 | β=— | — |
| ODP2_GEOSE_Tsuboyama_2023_1W4G | — | — | ρ=+0.2211 | β=— | — |
| OPSD_HUMAN_Wan_2019 | — | — | ρ=+0.5520 | β=— | — |
| OTC_HUMAN_Lo_2023 | — | — | ρ=+0.5309 | β=— | — |
| OTU7A_HUMAN_Tsuboyama_2023_2L2D | — | — | ρ=+0.3195 | β=— | — |
| OXDA_RHOTO_Vanella_2023_activity | — | — | ρ=+0.3702 | β=— | — |
| OXDA_RHOTO_Vanella_2023_expression | — | — | ρ=+0.3150 | β=— | — |
| P53_HUMAN_Giacomelli_2018_Null_Etoposide | — | — | ρ=+0.4323 | β=— | — |
| P53_HUMAN_Giacomelli_2018_Null_Nutlin | — | — | ρ=+0.4043 | β=— | — |
| P53_HUMAN_Giacomelli_2018_WT_Nutlin | — | — | ρ=+0.4002 | β=— | — |
| P53_HUMAN_Kotler_2018 | — | — | ρ=+0.6781 | β=— | — |
| P84126_THETH_Chan_2017 | — | — | ρ=+0.6047 | β=— | — |
| PA_I34A1_Wu_2015 | — | — | ρ=+0.0381 | β=— | — |
| PABP_YEAST_Melamed_2013 | — | — | ρ=+0.7158 | β=— | — |
| PAI1_HUMAN_Huttinger_2021 | — | — | ρ=+0.4479 | β=— | — |
| PHOT_CHLRE_Chen_2023 | — | — | ρ=+0.7491 | β=— | — |
| PIN1_HUMAN_Tsuboyama_2023_1I6C | — | — | ρ=+0.6699 | β=— | — |
| PITX2_HUMAN_Tsuboyama_2023_2L7M | — | — | ρ=+0.6262 | β=— | — |
| PKN1_HUMAN_Tsuboyama_2023_1URF | — | — | ρ=+0.2977 | β=— | — |
| POLG_CXB3N_Mattenberger_2021 | — | — | ρ=+0.3938 | β=— | — |
| POLG_DEN26_Suphatrakul_2023 | — | — | ρ=+0.1533 | β=— | — |
| POLG_HCVJF_Qi_2014 | — | — | ρ=+0.1307 | β=— | — |
| POLG_PESV_Tsuboyama_2023_2MXD | — | — | ρ=+0.1598 | β=— | — |
| PPARG_HUMAN_Majithia_2016 | — | — | ρ=+0.5937 | β=— | — |
| PPM1D_HUMAN_Miller_2022 | — | — | ρ=+0.6018 | β=— | — |
| PR40A_HUMAN_Tsuboyama_2023_1UZC | — | — | ρ=+0.8047 | β=— | — |
| PRKN_HUMAN_Clausen_2023 | — | — | ρ=+0.5012 | β=— | — |
| PSAE_PICP2_Tsuboyama_2023_1PSE | — | — | ρ=+0.7075 | β=— | — |
| PTEN_HUMAN_Matreyek_2021 | — | — | ρ=+0.4650 | β=— | — |
| PTEN_HUMAN_Mighell_2018 | — | — | ρ=+0.5185 | β=— | — |
| Q2N0S5_9HIV1_Haddox_2018 | — | — | ρ=+0.0442 | β=— | — |
| Q53Z42_HUMAN_McShan_2019_binding-TAPBPR | — | — | ρ=+0.3301 | β=— | — |
| Q53Z42_HUMAN_McShan_2019_expression | — | — | ρ=+0.5540 | β=— | — |
| Q59976_STRSQ_Romero_2015 | — | — | ρ=+0.5721 | β=— | — |
| Q6WV12_9MAXI_Somermeyer_2022 | — | — | ρ=+0.0096 | β=— | — |
| Q837P4_ENTFA_Meier_2023 | — | — | ρ=+0.5150 | β=— | — |
| Q837P5_ENTFA_Meier_2023 | — | — | ρ=+0.3738 | β=— | — |
| Q8WTC7_9CNID_Somermeyer_2022 | — | — | ρ=-0.0248 | β=— | — |
| R1AB_SARS2_Flynn_2022 | — | — | ρ=+0.1048 | β=— | — |
| RAD_ANTMA_Tsuboyama_2023_2CJJ | — | — | ρ=+0.4928 | β=— | — |
| RAF1_HUMAN_Zinkus-Boltz_2019 | — | — | ρ=+0.4733 | β=— | — |
| RASH_HUMAN_Bandaru_2017 | — | — | ρ=+0.4976 | β=— | — |
| RASK_HUMAN_Weng_2022_abundance | — | — | ρ=+0.2710 | β=— | — |
| RASK_HUMAN_Weng_2022_binding-DARPin_K55 | — | — | ρ=+0.6524 | β=— | — |
| RBP1_HUMAN_Tsuboyama_2023_2KWH | — | — | ρ=+0.5524 | β=— | — |
| RCD1_ARATH_Tsuboyama_2023_5OAO | — | — | ρ=+0.5049 | β=— | — |
| RCRO_LAMBD_Tsuboyama_2023_1ORC | — | — | ρ=+0.5958 | β=— | — |
| RD23A_HUMAN_Tsuboyama_2023_1IFY | — | — | ρ=+0.5127 | β=— | — |
| RDRP_I33A0_Li_2023 | — | — | ρ=+0.3218 | β=— | — |
| REV_HV1H2_Fernandes_2016 | — | — | ρ=+0.2404 | β=— | — |
| RFAH_ECOLI_Tsuboyama_2023_2LCL | — | — | ρ=+0.2991 | β=— | — |
| RL20_AQUAE_Tsuboyama_2023_1GYZ | — | — | ρ=+0.6989 | β=— | — |
| RL40A_YEAST_Mavor_2016 | — | — | ρ=+0.5179 | β=— | — |
| RL40A_YEAST_Roscoe_2013 | — | — | ρ=+0.5986 | β=— | — |
| RL40A_YEAST_Roscoe_2014 | — | — | ρ=+0.5300 | β=— | — |
| RNC_ECOLI_Weeks_2023 | — | — | ρ=+0.5962 | β=— | — |
| RPC1_BP434_Tsuboyama_2023_1R69 | — | — | ρ=+0.7084 | β=— | — |
| RPC1_LAMBD_Li_2019_high-expression | — | — | ρ=+0.5494 | β=— | — |
| RPC1_LAMBD_Li_2019_low-expression | — | — | ρ=+0.5231 | β=— | — |
| RS15_GEOSE_Tsuboyama_2023_1A32 | — | — | ρ=+0.4087 | β=— | — |
| S22A1_HUMAN_Yee_2023_abundance | — | — | ρ=+0.6264 | β=— | — |
| S22A1_HUMAN_Yee_2023_activity | — | — | ρ=+0.5752 | β=— | — |
| SAV1_MOUSE_Tsuboyama_2023_2YSB | — | — | ρ=+0.5120 | β=— | — |
| SBI_STAAM_Tsuboyama_2023_2JVG | — | — | ρ=+0.5726 | β=— | — |
| SC6A4_HUMAN_Young_2021 | — | — | ρ=+0.5426 | β=— | — |
| SCIN_STAAR_Tsuboyama_2023_2QFF | — | — | ρ=+0.3459 | β=— | — |
| SCN5A_HUMAN_Glazer_2019 | — | — | ρ=+0.1224 | β=— | — |
| SDA_BACSU_Tsuboyama_2023_1PV0 | — | — | ρ=+0.5943 | β=— | — |
| SERC_HUMAN_Xie_2023 | — | — | ρ=+0.5575 | β=— | — |
| SHOC2_HUMAN_Kwon_2022 | — | — | ρ=+0.4255 | β=— | — |
| SOX30_HUMAN_Tsuboyama_2023_7JJK | — | — | ρ=+0.3216 | β=— | — |
| SPA_STAAU_Tsuboyama_2023_1LP1 | — | — | ρ=+0.0053 | β=— | — |
| SPG1_STRSG_Olson_2014 | — | — | ρ=+0.3008 | β=— | — |
| SPG1_STRSG_Wu_2016 | — | — | ρ=+0.1283 | β=— | — |
| SPG2_STRSG_Tsuboyama_2023_5UBS | — | — | ρ=+0.4999 | β=— | — |
| SPIKE_SARS2_Starr_2020_binding | — | — | ρ=-0.0173 | β=— | — |
| SPIKE_SARS2_Starr_2020_expression | — | — | ρ=+0.0133 | β=— | — |
| SPTN1_CHICK_Tsuboyama_2023_1TUD | — | — | ρ=+0.6368 | β=— | — |
| SQSTM_MOUSE_Tsuboyama_2023_2RRU | — | — | ρ=+0.6180 | β=— | — |
| SR43C_ARATH_Tsuboyama_2023_2N88 | — | — | ρ=+0.6748 | β=— | — |
| SRBS1_HUMAN_Tsuboyama_2023_2O2W | — | — | ρ=+0.7339 | β=— | — |
| SRC_HUMAN_Ahler_2019 | — | — | ρ=+0.5551 | β=— | — |
| SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM | — | — | ρ=+0.4646 | β=— | — |
| SRC_HUMAN_Nguyen_2022 | — | — | ρ=+0.4707 | β=— | — |
| SUMO1_HUMAN_Weile_2017 | — | — | ρ=+0.5093 | β=— | — |
| SYUA_HUMAN_Newberry_2020 | — | — | ρ=+0.1373 | β=— | — |
| TADBP_HUMAN_Bolognesi_2019 | — | — | ρ=-0.1198 | β=— | — |
| TAT_HV1BR_Fernandes_2016 | — | — | ρ=+0.0174 | β=— | — |
| TCRG1_MOUSE_Tsuboyama_2023_1E0L | — | — | ρ=+0.7692 | β=— | — |
| THO1_YEAST_Tsuboyama_2023_2WQG | — | — | ρ=+0.5126 | β=— | — |
| TNKS2_HUMAN_Tsuboyama_2023_5JRT | — | — | ρ=+0.5252 | β=— | — |
| TPK1_HUMAN_Weile_2017 | — | — | ρ=+0.3386 | β=— | — |
| TPMT_HUMAN_Matreyek_2018 | — | — | ρ=+0.5387 | β=— | — |
| TPOR_HUMAN_Bridgford_2020 | — | — | ρ=+0.2899 | β=— | — |
| TRPC_SACS2_Chan_2017 | — | — | ρ=+0.6525 | β=— | — |
| TRPC_THEMA_Chan_2017 | — | — | ρ=+0.4637 | β=— | — |
| UBC9_HUMAN_Weile_2017 | — | — | ρ=+0.4726 | β=— | — |
| UBE4B_HUMAN_Tsuboyama_2023_3L1X | — | — | ρ=+0.4599 | β=— | — |
| UBE4B_MOUSE_Starita_2013 | — | — | ρ=+0.4584 | β=— | — |
| UBR5_HUMAN_Tsuboyama_2023_1I2T | — | — | ρ=+0.1969 | β=— | — |
| VG08_BPP22_Tsuboyama_2023_2GP8 | — | — | ρ=+0.6552 | β=— | — |
| VILI_CHICK_Tsuboyama_2023_1YU5 | — | — | ρ=+0.6617 | β=— | — |
| VKOR1_HUMAN_Chiasson_2020_abundance | — | — | ρ=+0.4933 | β=— | — |
| VKOR1_HUMAN_Chiasson_2020_activity | — | — | ρ=+0.4303 | β=— | — |
| VRPI_BPT7_Tsuboyama_2023_2WNM | — | — | ρ=+0.5875 | β=— | — |
| YAIA_ECOLI_Tsuboyama_2023_2KVT | — | — | ρ=+0.6245 | β=— | — |
| YAP1_HUMAN_Araya_2012 | — | — | ρ=+0.4661 | β=— | — |
| YNZC_BACSU_Tsuboyama_2023_2JVD | — | — | ρ=+0.6584 | β=— | — |


---

###  · run · `wt_marginals` on NVIDIA A100-SXM4-40GB

<!-- [run:wt_marginals:A0A140D2T1_ZIKV_Sourisseau_2019,A0A192B1T2_9HIV1_Haddox_2018,A0A1I9GEU1_NEIME_Kennouche_2019,A0A247D711_LISMN_Stadelmann_2021,A0A2Z5U3Z0_9INFA_Doud_2016,A0A2Z5U3Z0_9INFA_Wu_2014,A4D664_9INFA_Soh_2019,A4GRB6_PSEAI_Chen_2020,A4_HUMAN_Seuma_2022,AACC1_PSEAI_Dandage_2018,ACE2_HUMAN_Chan_2020,ADRB2_HUMAN_Jones_2020,AICDA_HUMAN_Gajula_2014_3cycles,AMFR_HUMAN_Tsuboyama_2023_4G3O,AMIE_PSEAE_Wrenbeck_2017,ANCSZ_Hobbs_2022,ARGR_ECOLI_Tsuboyama_2023_1AOY,B2L11_HUMAN_Dutta_2010_binding-Mcl-1,BBC1_YEAST_Tsuboyama_2023_1TG0,BCHB_CHLTE_Tsuboyama_2023_2KRU,BLAT_ECOLX_Deng_2012,BLAT_ECOLX_Firnberg_2014,BLAT_ECOLX_Jacquier_2013,BLAT_ECOLX_Stiffler_2015,BRCA1_HUMAN_Findlay_2018,BRCA2_HUMAN_Erwood_2022_HEK293T,C6KNH7_9INFA_Lee_2018,CALM1_HUMAN_Weile_2017,CAPSD_AAV2S_Sinai_2021,CAR11_HUMAN_Meitlis_2020_gof,CAR11_HUMAN_Meitlis_2020_lof,CAS9_STRP1_Spencer_2017_positive,CASP3_HUMAN_Roychowdhury_2020,CASP7_HUMAN_Roychowdhury_2020,CATR_CHLRE_Tsuboyama_2023_2AMI,CBPA2_HUMAN_Tsuboyama_2023_1O6X,CBS_HUMAN_Sun_2020,CBX4_HUMAN_Tsuboyama_2023_2K28,CCDB_ECOLI_Adkar_2012,CCDB_ECOLI_Tripathi_2016,CCR5_HUMAN_Gill_2023,CD19_HUMAN_Klesmith_2019_FMC_singles,CP2C9_HUMAN_Amorosi_2021_abundance,CP2C9_HUMAN_Amorosi_2021_activity,CSN4_MOUSE_Tsuboyama_2023_1UFM,CUE1_YEAST_Tsuboyama_2023_2MYX,D7PM05_CLYGR_Somermeyer_2022,DLG4_HUMAN_Faure_2021,DLG4_RAT_McLaughlin_2012,DN7A_SACS2_Tsuboyama_2023_1JIC,DNJA1_HUMAN_Tsuboyama_2023_2LO1,DOCK1_MOUSE_Tsuboyama_2023_2M0Y,DYR_ECOLI_Nguyen_2023,DYR_ECOLI_Thompson_2019,ENVZ_ECOLI_Ghose_2023,ENV_HV1B9_DuenasDecamp_2016,ENV_HV1BR_Haddox_2016,EPHB2_HUMAN_Tsuboyama_2023_1F0M,ERBB2_HUMAN_Elazar_2016,ESTA_BACSU_Nutschel_2020,F7YBW8_MESOW_Aakre_2015,F7YBW8_MESOW_Ding_2023,FECA_ECOLI_Tsuboyama_2023_2D1U,FKBP3_HUMAN_Tsuboyama_2023_2KFV,GAL4_YEAST_Kitzman_2015,GCN4_YEAST_Staller_2018,GDIA_HUMAN_Silverstein_2021,GFP_AEQVI_Sarkisyan_2016,GLPA_HUMAN_Elazar_2016,GRB2_HUMAN_Faure_2021,HCP_LAMBD_Tsuboyama_2023_2L6Q,HECD1_HUMAN_Tsuboyama_2023_3DKM,HEM3_HUMAN_Loggerenberg_2023,HIS7_YEAST_Pokusaeva_2019,HMDH_HUMAN_Jiang_2019,HSP82_YEAST_Cote-Hammarlof_2020_growth-H2O2,HSP82_YEAST_Flynn_2019,HSP82_YEAST_Mishra_2016,HXK4_HUMAN_Gersing_2022_activity,HXK4_HUMAN_Gersing_2023_abundance,I6TAH8_I68A0_Doud_2015,IF1_ECOLI_Kelsic_2016,ILF3_HUMAN_Tsuboyama_2023_2L33,ISDH_STAAW_Tsuboyama_2023_2LHR,KCNE1_HUMAN_Muhammad_2023_expression,KCNE1_HUMAN_Muhammad_2023_function,KCNH2_HUMAN_Kozek_2020,KCNJ2_MOUSE_Coyote-Maestas_2022_function,KCNJ2_MOUSE_Coyote-Maestas_2022_surface,KKA2_KLEPN_Melnikov_2014,LGK_LIPST_Klesmith_2015,LYAM1_HUMAN_Elazar_2016,MAFG_MOUSE_Tsuboyama_2023_1K1V,MBD11_ARATH_Tsuboyama_2023_6ACV,MET_HUMAN_Estevam_2023,MK01_HUMAN_Brenan_2016,MLAC_ECOLI_MacRae_2023,MSH2_HUMAN_Jia_2020,MTH3_HAEAE_RockahShmuel_2015,MTHR_HUMAN_Weile_2021,MYO3_YEAST_Tsuboyama_2023_2BTT,NCAP_I34A1_Doud_2015,NKX31_HUMAN_Tsuboyama_2023_2L9R,NPC1_HUMAN_Erwood_2022_HEK293T,NPC1_HUMAN_Erwood_2022_RPE1,NRAM_I33A0_Jiang_2016,NUD15_HUMAN_Suiter_2020,NUSA_ECOLI_Tsuboyama_2023_1WCL,NUSG_MYCTU_Tsuboyama_2023_2MI6,OBSCN_HUMAN_Tsuboyama_2023_1V1C,ODP2_GEOSE_Tsuboyama_2023_1W4G,OPSD_HUMAN_Wan_2019,OTC_HUMAN_Lo_2023,OTU7A_HUMAN_Tsuboyama_2023_2L2D,OXDA_RHOTO_Vanella_2023_activity,OXDA_RHOTO_Vanella_2023_expression,P53_HUMAN_Giacomelli_2018_Null_Etoposide,P53_HUMAN_Giacomelli_2018_Null_Nutlin,P53_HUMAN_Giacomelli_2018_WT_Nutlin,P53_HUMAN_Kotler_2018,P84126_THETH_Chan_2017,PABP_YEAST_Melamed_2013,PAI1_HUMAN_Huttinger_2021,PA_I34A1_Wu_2015,PHOT_CHLRE_Chen_2023,PIN1_HUMAN_Tsuboyama_2023_1I6C,PITX2_HUMAN_Tsuboyama_2023_2L7M,PKN1_HUMAN_Tsuboyama_2023_1URF,POLG_CXB3N_Mattenberger_2021,POLG_DEN26_Suphatrakul_2023,POLG_HCVJF_Qi_2014,POLG_PESV_Tsuboyama_2023_2MXD,PPARG_HUMAN_Majithia_2016,PPM1D_HUMAN_Miller_2022,PR40A_HUMAN_Tsuboyama_2023_1UZC,PRKN_HUMAN_Clausen_2023,PSAE_PICP2_Tsuboyama_2023_1PSE,PTEN_HUMAN_Matreyek_2021,PTEN_HUMAN_Mighell_2018,Q2N0S5_9HIV1_Haddox_2018,Q53Z42_HUMAN_McShan_2019_binding-TAPBPR,Q53Z42_HUMAN_McShan_2019_expression,Q59976_STRSQ_Romero_2015,Q6WV12_9MAXI_Somermeyer_2022,Q837P4_ENTFA_Meier_2023,Q837P5_ENTFA_Meier_2023,Q8WTC7_9CNID_Somermeyer_2022,R1AB_SARS2_Flynn_2022,RAD_ANTMA_Tsuboyama_2023_2CJJ,RAF1_HUMAN_Zinkus-Boltz_2019,RASH_HUMAN_Bandaru_2017,RASK_HUMAN_Weng_2022_abundance,RASK_HUMAN_Weng_2022_binding-DARPin_K55,RBP1_HUMAN_Tsuboyama_2023_2KWH,RCD1_ARATH_Tsuboyama_2023_5OAO,RCRO_LAMBD_Tsuboyama_2023_1ORC,RD23A_HUMAN_Tsuboyama_2023_1IFY,RDRP_I33A0_Li_2023,REV_HV1H2_Fernandes_2016,RFAH_ECOLI_Tsuboyama_2023_2LCL,RL20_AQUAE_Tsuboyama_2023_1GYZ,RL40A_YEAST_Mavor_2016,RL40A_YEAST_Roscoe_2013,RL40A_YEAST_Roscoe_2014,RNC_ECOLI_Weeks_2023,RPC1_BP434_Tsuboyama_2023_1R69,RPC1_LAMBD_Li_2019_high-expression,RPC1_LAMBD_Li_2019_low-expression,RS15_GEOSE_Tsuboyama_2023_1A32,S22A1_HUMAN_Yee_2023_abundance,S22A1_HUMAN_Yee_2023_activity,SAV1_MOUSE_Tsuboyama_2023_2YSB,SBI_STAAM_Tsuboyama_2023_2JVG,SC6A4_HUMAN_Young_2021,SCIN_STAAR_Tsuboyama_2023_2QFF,SCN5A_HUMAN_Glazer_2019,SDA_BACSU_Tsuboyama_2023_1PV0,SERC_HUMAN_Xie_2023,SHOC2_HUMAN_Kwon_2022,SOX30_HUMAN_Tsuboyama_2023_7JJK,SPA_STAAU_Tsuboyama_2023_1LP1,SPG1_STRSG_Olson_2014,SPG1_STRSG_Wu_2016,SPG2_STRSG_Tsuboyama_2023_5UBS,SPIKE_SARS2_Starr_2020_binding,SPIKE_SARS2_Starr_2020_expression,SPTN1_CHICK_Tsuboyama_2023_1TUD,SQSTM_MOUSE_Tsuboyama_2023_2RRU,SR43C_ARATH_Tsuboyama_2023_2N88,SRBS1_HUMAN_Tsuboyama_2023_2O2W,SRC_HUMAN_Ahler_2019,SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM,SRC_HUMAN_Nguyen_2022,SUMO1_HUMAN_Weile_2017,SYUA_HUMAN_Newberry_2020,TADBP_HUMAN_Bolognesi_2019,TAT_HV1BR_Fernandes_2016,TCRG1_MOUSE_Tsuboyama_2023_1E0L,THO1_YEAST_Tsuboyama_2023_2WQG,TNKS2_HUMAN_Tsuboyama_2023_5JRT,TPK1_HUMAN_Weile_2017,TPMT_HUMAN_Matreyek_2018,TPOR_HUMAN_Bridgford_2020,TRPC_SACS2_Chan_2017,TRPC_THEMA_Chan_2017,UBC9_HUMAN_Weile_2017,UBE4B_HUMAN_Tsuboyama_2023_3L1X,UBE4B_MOUSE_Starita_2013,UBR5_HUMAN_Tsuboyama_2023_1I2T,VG08_BPP22_Tsuboyama_2023_2GP8,VILI_CHICK_Tsuboyama_2023_1YU5,VKOR1_HUMAN_Chiasson_2020_abundance,VKOR1_HUMAN_Chiasson_2020_activity,VRPI_BPT7_Tsuboyama_2023_2WNM,YAIA_ECOLI_Tsuboyama_2023_2KVT,YAP1_HUMAN_Araya_2012,YNZC_BACSU_Tsuboyama_2023_2JVD] -->

**Method run: `wt_marginals` — 217 assays — mean ρ = +0.4441**

Wall total: 0.0s · Hardware: NVIDIA A100-SXM4-40GB · Integrity: Merkle `889c170fe685c443` · fair-esm@`2b369911`

| Assay | L | N | ρ | β (s/AA²) | Wall |
|---|---|---|---|---|---|
| A0A140D2T1_ZIKV_Sourisseau_2019 | — | — | ρ=+0.2087 | β=— | — |
| A0A192B1T2_9HIV1_Haddox_2018 | — | — | ρ=+0.0642 | β=— | — |
| A0A1I9GEU1_NEIME_Kennouche_2019 | — | — | ρ=+0.0315 | β=— | — |
| A0A247D711_LISMN_Stadelmann_2021 | — | — | ρ=+0.0656 | β=— | — |
| A0A2Z5U3Z0_9INFA_Doud_2016 | — | — | ρ=+0.4922 | β=— | — |
| A0A2Z5U3Z0_9INFA_Wu_2014 | — | — | ρ=+0.4522 | β=— | — |
| A4_HUMAN_Seuma_2022 | — | — | ρ=+0.4398 | β=— | — |
| A4D664_9INFA_Soh_2019 | — | — | ρ=+0.1370 | β=— | — |
| A4GRB6_PSEAI_Chen_2020 | — | — | ρ=+0.7281 | β=— | — |
| AACC1_PSEAI_Dandage_2018 | — | — | ρ=+0.4762 | β=— | — |
| ACE2_HUMAN_Chan_2020 | — | — | ρ=+0.2297 | β=— | — |
| ADRB2_HUMAN_Jones_2020 | — | — | ρ=+0.4971 | β=— | — |
| AICDA_HUMAN_Gajula_2014_3cycles | — | — | ρ=+0.2898 | β=— | — |
| AMFR_HUMAN_Tsuboyama_2023_4G3O | — | — | ρ=+0.2737 | β=— | — |
| AMIE_PSEAE_Wrenbeck_2017 | — | — | ρ=+0.5533 | β=— | — |
| ANCSZ_Hobbs_2022 | — | — | ρ=+0.5948 | β=— | — |
| ARGR_ECOLI_Tsuboyama_2023_1AOY | — | — | ρ=+0.5101 | β=— | — |
| B2L11_HUMAN_Dutta_2010_binding-Mcl-1 | — | — | ρ=+0.3215 | β=— | — |
| BBC1_YEAST_Tsuboyama_2023_1TG0 | — | — | ρ=+0.4737 | β=— | — |
| BCHB_CHLTE_Tsuboyama_2023_2KRU | — | — | ρ=+0.5409 | β=— | — |
| BLAT_ECOLX_Deng_2012 | — | — | ρ=+0.5291 | β=— | — |
| BLAT_ECOLX_Firnberg_2014 | — | — | ρ=+0.7279 | β=— | — |
| BLAT_ECOLX_Jacquier_2013 | — | — | ρ=+0.6933 | β=— | — |
| BLAT_ECOLX_Stiffler_2015 | — | — | ρ=+0.7227 | β=— | — |
| BRCA1_HUMAN_Findlay_2018 | — | — | ρ=+0.5137 | β=— | — |
| BRCA2_HUMAN_Erwood_2022_HEK293T | — | — | ρ=+0.4836 | β=— | — |
| C6KNH7_9INFA_Lee_2018 | — | — | ρ=+0.4636 | β=— | — |
| CALM1_HUMAN_Weile_2017 | — | — | ρ=+0.2100 | β=— | — |
| CAPSD_AAV2S_Sinai_2021 | — | — | ρ=+0.1999 | β=— | — |
| CAR11_HUMAN_Meitlis_2020_gof | — | — | ρ=+0.3284 | β=— | — |
| CAR11_HUMAN_Meitlis_2020_lof | — | — | ρ=+0.5003 | β=— | — |
| CAS9_STRP1_Spencer_2017_positive | — | — | ρ=+0.1839 | β=— | — |
| CASP3_HUMAN_Roychowdhury_2020 | — | — | ρ=+0.6326 | β=— | — |
| CASP7_HUMAN_Roychowdhury_2020 | — | — | ρ=+0.6139 | β=— | — |
| CATR_CHLRE_Tsuboyama_2023_2AMI | — | — | ρ=+0.6601 | β=— | — |
| CBPA2_HUMAN_Tsuboyama_2023_1O6X | — | — | ρ=+0.7622 | β=— | — |
| CBS_HUMAN_Sun_2020 | — | — | ρ=+0.3368 | β=— | — |
| CBX4_HUMAN_Tsuboyama_2023_2K28 | — | — | ρ=+0.6709 | β=— | — |
| CCDB_ECOLI_Adkar_2012 | — | — | ρ=+0.4783 | β=— | — |
| CCDB_ECOLI_Tripathi_2016 | — | — | ρ=+0.5141 | β=— | — |
| CCR5_HUMAN_Gill_2023 | — | — | ρ=+0.3448 | β=— | — |
| CD19_HUMAN_Klesmith_2019_FMC_singles | — | — | ρ=+0.0527 | β=— | — |
| CP2C9_HUMAN_Amorosi_2021_abundance | — | — | ρ=+0.6279 | β=— | — |
| CP2C9_HUMAN_Amorosi_2021_activity | — | — | ρ=+0.6720 | β=— | — |
| CSN4_MOUSE_Tsuboyama_2023_1UFM | — | — | ρ=+0.5214 | β=— | — |
| CUE1_YEAST_Tsuboyama_2023_2MYX | — | — | ρ=+0.5170 | β=— | — |
| D7PM05_CLYGR_Somermeyer_2022 | — | — | ρ=+0.4473 | β=— | — |
| DLG4_HUMAN_Faure_2021 | — | — | ρ=+0.5206 | β=— | — |
| DLG4_RAT_McLaughlin_2012 | — | — | ρ=+0.4391 | β=— | — |
| DN7A_SACS2_Tsuboyama_2023_1JIC | — | — | ρ=+0.3195 | β=— | — |
| DNJA1_HUMAN_Tsuboyama_2023_2LO1 | — | — | ρ=+0.7912 | β=— | — |
| DOCK1_MOUSE_Tsuboyama_2023_2M0Y | — | — | ρ=+0.5191 | β=— | — |
| DYR_ECOLI_Nguyen_2023 | — | — | ρ=+0.5384 | β=— | — |
| DYR_ECOLI_Thompson_2019 | — | — | ρ=+0.4793 | β=— | — |
| ENV_HV1B9_DuenasDecamp_2016 | — | — | ρ=+0.0422 | β=— | — |
| ENV_HV1BR_Haddox_2016 | — | — | ρ=+0.0359 | β=— | — |
| ENVZ_ECOLI_Ghose_2023 | — | — | ρ=+0.1733 | β=— | — |
| EPHB2_HUMAN_Tsuboyama_2023_1F0M | — | — | ρ=+0.8147 | β=— | — |
| ERBB2_HUMAN_Elazar_2016 | — | — | ρ=+0.4268 | β=— | — |
| ESTA_BACSU_Nutschel_2020 | — | — | ρ=+0.3002 | β=— | — |
| F7YBW8_MESOW_Ding_2023 | — | — | ρ=+0.4933 | β=— | — |
| F7YBW8_MESOW_Aakre_2015 | — | — | ρ=+0.4160 | β=— | — |
| FECA_ECOLI_Tsuboyama_2023_2D1U | — | — | ρ=+0.4970 | β=— | — |
| FKBP3_HUMAN_Tsuboyama_2023_2KFV | — | — | ρ=+0.2176 | β=— | — |
| GAL4_YEAST_Kitzman_2015 | — | — | ρ=+0.6558 | β=— | — |
| GCN4_YEAST_Staller_2018 | — | — | ρ=+0.2619 | β=— | — |
| GDIA_HUMAN_Silverstein_2021 | — | — | ρ=+0.3784 | β=— | — |
| GFP_AEQVI_Sarkisyan_2016 | — | — | ρ=+0.5256 | β=— | — |
| GLPA_HUMAN_Elazar_2016 | — | — | ρ=+0.4065 | β=— | — |
| GRB2_HUMAN_Faure_2021 | — | — | ρ=+0.6477 | β=— | — |
| HCP_LAMBD_Tsuboyama_2023_2L6Q | — | — | ρ=+0.6955 | β=— | — |
| HECD1_HUMAN_Tsuboyama_2023_3DKM | — | — | ρ=+0.3899 | β=— | — |
| HEM3_HUMAN_Loggerenberg_2023 | — | — | ρ=+0.3847 | β=— | — |
| HIS7_YEAST_Pokusaeva_2019 | — | — | ρ=+0.4197 | β=— | — |
| HMDH_HUMAN_Jiang_2019 | — | — | ρ=+0.5042 | β=— | — |
| HSP82_YEAST_Cote-Hammarlof_2020_growth-H2O2 | — | — | ρ=+0.2401 | β=— | — |
| HSP82_YEAST_Flynn_2019 | — | — | ρ=+0.2760 | β=— | — |
| HSP82_YEAST_Mishra_2016 | — | — | ρ=+0.4359 | β=— | — |
| HXK4_HUMAN_Gersing_2022_activity | — | — | ρ=+0.5121 | β=— | — |
| HXK4_HUMAN_Gersing_2023_abundance | — | — | ρ=+0.3720 | β=— | — |
| I6TAH8_I68A0_Doud_2015 | — | — | ρ=+0.0168 | β=— | — |
| IF1_ECOLI_Kelsic_2016 | — | — | ρ=+0.5794 | β=— | — |
| ILF3_HUMAN_Tsuboyama_2023_2L33 | — | — | ρ=+0.2802 | β=— | — |
| ISDH_STAAW_Tsuboyama_2023_2LHR | — | — | ρ=+0.4297 | β=— | — |
| KCNE1_HUMAN_Muhammad_2023_expression | — | — | ρ=+0.1017 | β=— | — |
| KCNE1_HUMAN_Muhammad_2023_function | — | — | ρ=+0.5273 | β=— | — |
| KCNH2_HUMAN_Kozek_2020 | — | — | ρ=+0.5202 | β=— | — |
| KCNJ2_MOUSE_Coyote-Maestas_2022_function | — | — | ρ=+0.3714 | β=— | — |
| KCNJ2_MOUSE_Coyote-Maestas_2022_surface | — | — | ρ=+0.3454 | β=— | — |
| KKA2_KLEPN_Melnikov_2014 | — | — | ρ=+0.5793 | β=— | — |
| LGK_LIPST_Klesmith_2015 | — | — | ρ=+0.5164 | β=— | — |
| LYAM1_HUMAN_Elazar_2016 | — | — | ρ=+0.3212 | β=— | — |
| MAFG_MOUSE_Tsuboyama_2023_1K1V | — | — | ρ=+0.6192 | β=— | — |
| MBD11_ARATH_Tsuboyama_2023_6ACV | — | — | ρ=+0.7400 | β=— | — |
| MET_HUMAN_Estevam_2023 | — | — | ρ=+0.5885 | β=— | — |
| MK01_HUMAN_Brenan_2016 | — | — | ρ=+0.1639 | β=— | — |
| MLAC_ECOLI_MacRae_2023 | — | — | ρ=+0.3660 | β=— | — |
| MSH2_HUMAN_Jia_2020 | — | — | ρ=+0.3187 | β=— | — |
| MTH3_HAEAE_RockahShmuel_2015 | — | — | ρ=+0.5018 | β=— | — |
| MTHR_HUMAN_Weile_2021 | — | — | ρ=+0.3222 | β=— | — |
| MYO3_YEAST_Tsuboyama_2023_2BTT | — | — | ρ=+0.5452 | β=— | — |
| NCAP_I34A1_Doud_2015 | — | — | ρ=+0.0200 | β=— | — |
| NKX31_HUMAN_Tsuboyama_2023_2L9R | — | — | ρ=+0.6714 | β=— | — |
| NPC1_HUMAN_Erwood_2022_HEK293T | — | — | ρ=+0.7011 | β=— | — |
| NPC1_HUMAN_Erwood_2022_RPE1 | — | — | ρ=+0.6933 | β=— | — |
| NRAM_I33A0_Jiang_2016 | — | — | ρ=+0.1663 | β=— | — |
| NUD15_HUMAN_Suiter_2020 | — | — | ρ=+0.5281 | β=— | — |
| NUSA_ECOLI_Tsuboyama_2023_1WCL | — | — | ρ=+0.6317 | β=— | — |
| NUSG_MYCTU_Tsuboyama_2023_2MI6 | — | — | ρ=+0.4837 | β=— | — |
| OBSCN_HUMAN_Tsuboyama_2023_1V1C | — | — | ρ=+0.7795 | β=— | — |
| ODP2_GEOSE_Tsuboyama_2023_1W4G | — | — | ρ=+0.0544 | β=— | — |
| OPSD_HUMAN_Wan_2019 | — | — | ρ=+0.4766 | β=— | — |
| OTC_HUMAN_Lo_2023 | — | — | ρ=+0.5229 | β=— | — |
| OTU7A_HUMAN_Tsuboyama_2023_2L2D | — | — | ρ=+0.3338 | β=— | — |
| OXDA_RHOTO_Vanella_2023_activity | — | — | ρ=+0.3544 | β=— | — |
| OXDA_RHOTO_Vanella_2023_expression | — | — | ρ=+0.2987 | β=— | — |
| P53_HUMAN_Giacomelli_2018_Null_Etoposide | — | — | ρ=+0.4243 | β=— | — |
| P53_HUMAN_Giacomelli_2018_Null_Nutlin | — | — | ρ=+0.3969 | β=— | — |
| P53_HUMAN_Giacomelli_2018_WT_Nutlin | — | — | ρ=+0.3843 | β=— | — |
| P53_HUMAN_Kotler_2018 | — | — | ρ=+0.6624 | β=— | — |
| P84126_THETH_Chan_2017 | — | — | ρ=+0.5869 | β=— | — |
| PA_I34A1_Wu_2015 | — | — | ρ=+0.0385 | β=— | — |
| PABP_YEAST_Melamed_2013 | — | — | ρ=+0.6821 | β=— | — |
| PAI1_HUMAN_Huttinger_2021 | — | — | ρ=+0.4444 | β=— | — |
| PHOT_CHLRE_Chen_2023 | — | — | ρ=+0.5995 | β=— | — |
| PIN1_HUMAN_Tsuboyama_2023_1I6C | — | — | ρ=+0.6865 | β=— | — |
| PITX2_HUMAN_Tsuboyama_2023_2L7M | — | — | ρ=+0.6024 | β=— | — |
| PKN1_HUMAN_Tsuboyama_2023_1URF | — | — | ρ=+0.3053 | β=— | — |
| POLG_CXB3N_Mattenberger_2021 | — | — | ρ=+0.3814 | β=— | — |
| POLG_DEN26_Suphatrakul_2023 | — | — | ρ=+0.1426 | β=— | — |
| POLG_HCVJF_Qi_2014 | — | — | ρ=+0.1336 | β=— | — |
| POLG_PESV_Tsuboyama_2023_2MXD | — | — | ρ=+0.4059 | β=— | — |
| PPARG_HUMAN_Majithia_2016 | — | — | ρ=+0.5900 | β=— | — |
| PPM1D_HUMAN_Miller_2022 | — | — | ρ=+0.5933 | β=— | — |
| PR40A_HUMAN_Tsuboyama_2023_1UZC | — | — | ρ=+0.7994 | β=— | — |
| PRKN_HUMAN_Clausen_2023 | — | — | ρ=+0.5029 | β=— | — |
| PSAE_PICP2_Tsuboyama_2023_1PSE | — | — | ρ=+0.7020 | β=— | — |
| PTEN_HUMAN_Matreyek_2021 | — | — | ρ=+0.4644 | β=— | — |
| PTEN_HUMAN_Mighell_2018 | — | — | ρ=+0.4927 | β=— | — |
| Q2N0S5_9HIV1_Haddox_2018 | — | — | ρ=+0.0283 | β=— | — |
| Q53Z42_HUMAN_McShan_2019_binding-TAPBPR | — | — | ρ=+0.3219 | β=— | — |
| Q53Z42_HUMAN_McShan_2019_expression | — | — | ρ=+0.5596 | β=— | — |
| Q59976_STRSQ_Romero_2015 | — | — | ρ=+0.5546 | β=— | — |
| Q6WV12_9MAXI_Somermeyer_2022 | — | — | ρ=+0.2384 | β=— | — |
| Q837P4_ENTFA_Meier_2023 | — | — | ρ=+0.5109 | β=— | — |
| Q837P5_ENTFA_Meier_2023 | — | — | ρ=+0.3900 | β=— | — |
| Q8WTC7_9CNID_Somermeyer_2022 | — | — | ρ=+0.2231 | β=— | — |
| R1AB_SARS2_Flynn_2022 | — | — | ρ=+0.1180 | β=— | — |
| RAD_ANTMA_Tsuboyama_2023_2CJJ | — | — | ρ=+0.5176 | β=— | — |
| RAF1_HUMAN_Zinkus-Boltz_2019 | — | — | ρ=+0.4490 | β=— | — |
| RASH_HUMAN_Bandaru_2017 | — | — | ρ=+0.4769 | β=— | — |
| RASK_HUMAN_Weng_2022_abundance | — | — | ρ=+0.3113 | β=— | — |
| RASK_HUMAN_Weng_2022_binding-DARPin_K55 | — | — | ρ=+0.6177 | β=— | — |
| RBP1_HUMAN_Tsuboyama_2023_2KWH | — | — | ρ=+0.5133 | β=— | — |
| RCD1_ARATH_Tsuboyama_2023_5OAO | — | — | ρ=+0.5487 | β=— | — |
| RCRO_LAMBD_Tsuboyama_2023_1ORC | — | — | ρ=+0.5658 | β=— | — |
| RD23A_HUMAN_Tsuboyama_2023_1IFY | — | — | ρ=+0.4695 | β=— | — |
| RDRP_I33A0_Li_2023 | — | — | ρ=+0.2904 | β=— | — |
| REV_HV1H2_Fernandes_2016 | — | — | ρ=+0.2265 | β=— | — |
| RFAH_ECOLI_Tsuboyama_2023_2LCL | — | — | ρ=+0.2959 | β=— | — |
| RL20_AQUAE_Tsuboyama_2023_1GYZ | — | — | ρ=+0.7216 | β=— | — |
| RL40A_YEAST_Mavor_2016 | — | — | ρ=+0.5246 | β=— | — |
| RL40A_YEAST_Roscoe_2013 | — | — | ρ=+0.5956 | β=— | — |
| RL40A_YEAST_Roscoe_2014 | — | — | ρ=+0.5280 | β=— | — |
| RNC_ECOLI_Weeks_2023 | — | — | ρ=+0.5835 | β=— | — |
| RPC1_BP434_Tsuboyama_2023_1R69 | — | — | ρ=+0.7046 | β=— | — |
| RPC1_LAMBD_Li_2019_high-expression | — | — | ρ=+0.5346 | β=— | — |
| RPC1_LAMBD_Li_2019_low-expression | — | — | ρ=+0.4947 | β=— | — |
| RS15_GEOSE_Tsuboyama_2023_1A32 | — | — | ρ=+0.4166 | β=— | — |
| S22A1_HUMAN_Yee_2023_abundance | — | — | ρ=+0.6183 | β=— | — |
| S22A1_HUMAN_Yee_2023_activity | — | — | ρ=+0.5597 | β=— | — |
| SAV1_MOUSE_Tsuboyama_2023_2YSB | — | — | ρ=+0.4742 | β=— | — |
| SBI_STAAM_Tsuboyama_2023_2JVG | — | — | ρ=+0.5221 | β=— | — |
| SC6A4_HUMAN_Young_2021 | — | — | ρ=+0.5420 | β=— | — |
| SCIN_STAAR_Tsuboyama_2023_2QFF | — | — | ρ=+0.3774 | β=— | — |
| SCN5A_HUMAN_Glazer_2019 | — | — | ρ=+0.1214 | β=— | — |
| SDA_BACSU_Tsuboyama_2023_1PV0 | — | — | ρ=+0.6044 | β=— | — |
| SERC_HUMAN_Xie_2023 | — | — | ρ=+0.5483 | β=— | — |
| SHOC2_HUMAN_Kwon_2022 | — | — | ρ=+0.4073 | β=— | — |
| SOX30_HUMAN_Tsuboyama_2023_7JJK | — | — | ρ=+0.3327 | β=— | — |
| SPA_STAAU_Tsuboyama_2023_1LP1 | — | — | ρ=+0.3949 | β=— | — |
| SPG1_STRSG_Olson_2014 | — | — | ρ=+0.3186 | β=— | — |
| SPG1_STRSG_Wu_2016 | — | — | ρ=+0.1374 | β=— | — |
| SPG2_STRSG_Tsuboyama_2023_5UBS | — | — | ρ=+0.5450 | β=— | — |
| SPIKE_SARS2_Starr_2020_binding | — | — | ρ=-0.0200 | β=— | — |
| SPIKE_SARS2_Starr_2020_expression | — | — | ρ=+0.0184 | β=— | — |
| SPTN1_CHICK_Tsuboyama_2023_1TUD | — | — | ρ=+0.6734 | β=— | — |
| SQSTM_MOUSE_Tsuboyama_2023_2RRU | — | — | ρ=+0.6376 | β=— | — |
| SR43C_ARATH_Tsuboyama_2023_2N88 | — | — | ρ=+0.6758 | β=— | — |
| SRBS1_HUMAN_Tsuboyama_2023_2O2W | — | — | ρ=+0.7447 | β=— | — |
| SRC_HUMAN_Ahler_2019 | — | — | ρ=+0.5472 | β=— | — |
| SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM | — | — | ρ=+0.4557 | β=— | — |
| SRC_HUMAN_Nguyen_2022 | — | — | ρ=+0.4587 | β=— | — |
| SUMO1_HUMAN_Weile_2017 | — | — | ρ=+0.5008 | β=— | — |
| SYUA_HUMAN_Newberry_2020 | — | — | ρ=+0.1280 | β=— | — |
| TADBP_HUMAN_Bolognesi_2019 | — | — | ρ=-0.1187 | β=— | — |
| TAT_HV1BR_Fernandes_2016 | — | — | ρ=-0.0447 | β=— | — |
| TCRG1_MOUSE_Tsuboyama_2023_1E0L | — | — | ρ=+0.7552 | β=— | — |
| THO1_YEAST_Tsuboyama_2023_2WQG | — | — | ρ=+0.5267 | β=— | — |
| TNKS2_HUMAN_Tsuboyama_2023_5JRT | — | — | ρ=+0.5228 | β=— | — |
| TPK1_HUMAN_Weile_2017 | — | — | ρ=+0.3334 | β=— | — |
| TPMT_HUMAN_Matreyek_2018 | — | — | ρ=+0.5371 | β=— | — |
| TPOR_HUMAN_Bridgford_2020 | — | — | ρ=+0.2949 | β=— | — |
| TRPC_SACS2_Chan_2017 | — | — | ρ=+0.6441 | β=— | — |
| TRPC_THEMA_Chan_2017 | — | — | ρ=+0.4592 | β=— | — |
| UBC9_HUMAN_Weile_2017 | — | — | ρ=+0.4546 | β=— | — |
| UBE4B_HUMAN_Tsuboyama_2023_3L1X | — | — | ρ=+0.4489 | β=— | — |
| UBE4B_MOUSE_Starita_2013 | — | — | ρ=+0.4138 | β=— | — |
| UBR5_HUMAN_Tsuboyama_2023_1I2T | — | — | ρ=+0.5532 | β=— | — |
| VG08_BPP22_Tsuboyama_2023_2GP8 | — | — | ρ=+0.6618 | β=— | — |
| VILI_CHICK_Tsuboyama_2023_1YU5 | — | — | ρ=+0.6605 | β=— | — |
| VKOR1_HUMAN_Chiasson_2020_abundance | — | — | ρ=+0.5033 | β=— | — |
| VKOR1_HUMAN_Chiasson_2020_activity | — | — | ρ=+0.4131 | β=— | — |
| VRPI_BPT7_Tsuboyama_2023_2WNM | — | — | ρ=+0.5756 | β=— | — |
| YAIA_ECOLI_Tsuboyama_2023_2KVT | — | — | ρ=+0.6149 | β=— | — |
| YAP1_HUMAN_Araya_2012 | — | — | ρ=+0.4227 | β=— | — |
| YNZC_BACSU_Tsuboyama_2023_2JVD | — | — | ρ=+0.6778 | β=— | — |


---

### 2026-09-20 · `4da2fbe` · nialloleary

**fix(image_c): remove biopython==1.79 pin — esm SDK needs >=1.80 for Bio.Data.PDBData**

Both source implementations pinned as git submodules with exact SHAs: fair-esm @ 2b369911 (the commit that produced ρ=0.414) and ProteinGym @ 144fe22b. Reproducibility gate added: fair-esm fp32 vs transformers fp16 on SNCA, ρ must be ≥ 0.999.

| | |
|---|---|
| SHA | `4da2fbe0cfa12b8030e213f90a814ffae6d20ba4` |
| Changed | 1 file changed, 1 insertion(+), 1 deletion(-) |
| Files | `modal_app.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 4da2fbe --no-edit` |

---

### 2026-09-20 · `b9e6d6f` · nialloleary

**fix: add httpx to image_c (esm.sdk.forge hard import)**

fix: add httpx to image_c (esm.sdk.forge hard import)

| | |
|---|---|
| SHA | `b9e6d6f6f3a07c4af50078c01ce6da51e83f5820` |
| Changed | 1 file changed, 1 insertion(+) |
| Files | `modal_app.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert b9e6d6f --no-edit` |

---

### 2026-09-20 · `3474954` · nialloleary

**results: add Track A benchmark data (217 assays, fair-esm A100)**

Corrected all entrypoints to target A100 40GB — the GPU used in the original paper. MPS backend kept for local smoke tests only.

| | |
|---|---|
| SHA | `3474954f8cd771b64f325d693812cf46440737b5` |
| Changed | 4 files changed, 1412 insertions(+) |
| Files | `results/comparison_stream.jsonl`  `results/modal_log_pulled.md`  `results/ref_proteingym_stream.jsonl`  `results/track_a_summary.md` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 3474954 --no-edit` |

---

### 2026-09-20 · `b2b228e` · nialloleary

**docs: add inference cost analysis (from protein-language-model-experiments)**

docs: add inference cost analysis (from protein-language-model-experiments)

| | |
|---|---|
| SHA | `b2b228e7989a9c56a143164230b5b350e6bd0456` |
| Changed | 1 file changed, 110 insertions(+) |
| Files | `docs/inference_cost_analysis.md` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert b2b228e --no-edit` |

---

### 2026-09-20 · `3c5780c` · nialloleary

**legacy: update README with migration map for protein-language-model-experiments**

legacy: update README with migration map for protein-language-model-experiments

| | |
|---|---|
| SHA | `3c5780cae3cd84479fff115e755f8303f0e67ec4` |
| Changed | 1 file changed, 14 insertions(+), 1 deletion(-) |
| Files | `legacy/protein-language-model-experiments/README.md` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 3c5780c --no-edit` |

---

### 2026-09-20 · `8d024ec` · nialloleary

**feat(esmc-300m): local MPS inference + refs entry**

feat(esmc-300m): local MPS inference + refs entry

| | |
|---|---|
| SHA | `8d024ece97787706927441e3333c881723943f63` |
| Changed | 2 files changed, 121 insertions(+), 1 deletion(-) |
| Files | `refs/README.md`  `scoring/esmc/local_300m.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 8d024ec --no-edit` |

---

### 2026-09-20 · `037b20b` · nialloleary

**fix(esmc-300m): correct flash_attn/MPS comment, add ρ target**

fix(esmc-300m): correct flash_attn/MPS comment, add ρ target

| | |
|---|---|
| SHA | `037b20bb886d39cc0ecf0fe6beeac1fd7b1de0ec` |
| Changed | 2 files changed, 11 insertions(+), 3 deletions(-) |
| Files | `refs/README.md`  `scoring/esmc/local_300m.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 037b20b --no-edit` |

---

### 2026-09-20 · `1aed4a1` · nialloleary

**Add native MLX implementation of ESMC-300M/600M for Apple Silicon**

Add native MLX implementation of ESMC-300M/600M for Apple Silicon

| | |
|---|---|
| SHA | `1aed4a1669954bc54330ca241f58475e12266d4a` |
| Changed | 1 file changed, 326 insertions(+) |
| Files | `scoring/esmc/mlx_esmc.py` |
| Integrity | Merkle `889c170fe685c443` · fair-esm@`2b369911` · ProteinGym@`144fe22b` |
| Undo | `git revert 1aed4a1 --no-edit` |
