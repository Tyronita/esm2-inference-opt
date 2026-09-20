# Implementation Provenance

Every implementation of ESM-2 masked-marginal scoring that this repo is
compared against, with exact URLs pinned to the lines that matter.

---

## 1. fair-esm — Meta's original library

**Submodule:** `refs/fair-esm/`  
**Repo:** https://github.com/facebookresearch/esm  
**Commit pinned:** see `.gitmodules`  
**Install:** `pip install fair-esm`  
**HuggingFace hub entry:** https://huggingface.co/facebook/esm2_t33_650M_UR50D  

Key files:
| File | URL | What it does |
|---|---|---|
| `esm/pretrained.py:374` | https://github.com/facebookresearch/esm/blob/main/esm/pretrained.py#L374 | `esm2_t33_650M_UR50D()` — loads model + alphabet |
| `esm/model/esm2.py` | https://github.com/facebookresearch/esm/blob/main/esm/model/esm2.py | Full ESM-2 architecture (RoPE, pre-norm, GELU) |
| `esm/modules.py` | https://github.com/facebookresearch/esm/blob/main/esm/modules.py | `TransformerLayer` — `need_weights=True` hardcoded |

**Critical note:** `TransformerLayer.forward` has `need_weights=True` hardcoded,
forcing explicit `[B, H, T, T]` attention materialisation on every pass. This is
the slow path that produced the published ρ=0.414. Our transformers implementation
uses `attn_implementation="sdpa"` — numerically equivalent but faster.

---

## 2. ProteinGym — canonical scoring script

**Submodule:** `refs/ProteinGym/`  
**Repo:** https://github.com/OATML-Markslab/ProteinGym  
**Commit pinned:** see `.gitmodules`  

Key files:
| File | URL | What it does |
|---|---|---|
| `proteingym/baselines/esm/compute_fitness.py` | https://github.com/OATML-Markslab/ProteinGym/blob/main/proteingym/baselines/esm/compute_fitness.py | Scoring entry point — uses fair-esm, `--scoring-strategy masked-marginals` |
| `proteingym/utils/scoring_utils.py:1` | https://github.com/OATML-Markslab/ProteinGym/blob/main/proteingym/utils/scoring_utils.py#L1 | `get_optimal_window` — windowing for L>1022 |
| `reference_files/DMS_substitutions.csv` | https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv | 217-assay metadata, `target_seq`, filenames |
| `benchmarks/DMS_zero_shot/substitutions/Spearman/` | https://github.com/OATML-Markslab/ProteinGym/tree/main/benchmarks/DMS_zero_shot/substitutions/Spearman | Per-assay published Spearman scores — validate against these |

**Published baseline:** ESM-2 650M, masked-marginals, 217 assays: **ρ = 0.414 ± 0.012**  
Source: Notin et al. NeurIPS 2023. Table reported with fair-esm + float32 + optimal windowing.

---

## 3. HuggingFace transformers — our implementation

**Package:** `transformers==4.44.0`  
**Model hub:** https://huggingface.co/facebook/esm2_t33_650M_UR50D  
**Model card:** https://huggingface.co/facebook/esm2_t33_650M_UR50D/blob/main/README.md  
**Config:** https://huggingface.co/facebook/esm2_t33_650M_UR50D/blob/main/config.json  
**transformers ESM source:** https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/modeling_esm.py  

Loaded in our code as:
```python
from transformers import AutoTokenizer, EsmForMaskedLM
tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", torch_dtype=torch.float16)
```

---

## 4. Our scoring implementations

All in `scoring/`:

| Module | Passes/assay | Source | Deviation from original |
|---|---|---|---|
| `wt_marginals.py` | 1 | Meier et al. 2021 | transformers not fair-esm; fp16 |
| `masked_marginals.py` | L | Meier et al. 2021 / ProteinGym | transformers; fp16; window ±1 char off for L>1022 |
| `pseudo_ppl.py` | L (mutant) | Meier et al. 2021 | transformers; fp16 |
| `batched_masked.py` | ceil(L/B) | This work | approximation; not in published benchmark |
| `simple_ofs.py` | 1 | NOT Kantroo 2024 | functionally = wt_marginals |

---

## 5. ESMC 600M — EvolutionaryScale SDK

**Submodule:** `refs/evolutionaryscale-esm/`
**Repo:** https://github.com/evolutionaryscale/esm
**Commit pinned:** 43b4548b86762edfa747b07d5f440aad3c33acee (v3.0.7)
**Install:** `pip install "esm>=3.0.0"`
**HuggingFace hub:** https://huggingface.co/biohub/ESMC-600M
**License:** MIT

Key files:
| File | What it does |
|---|---|
| `esm/models/esmc/model.py` | `ESMC` (SDK client) and `EsmcForMaskedLM` (HF-style) |
| `esm/models/esmc/tokenizer.py` | `EsmcTokenizer` — vocab: 64 tokens, AAs at 4-23, `<mask>`=32 |
| `esm/models/esmc/config.py` | `EsmcConfig` — ESMC_EXPANSION_RATIO=8/3, SwiGLU FFN |
| `cookbook/snippets/esmc.py` | Canonical usage — encode/logits SDK + raw forward |
| `esm/sdk/api.py` | `ESMProtein`, `LogitsConfig`, `LogitsOutput` |

**ProteinGym ESMC baseline:**
`refs/ProteinGym/proteingym/baselines/evoscale/compute_fitness.py` — canonical ESMC scoring.
Score function: `_score_mutations_common` — mask each position, get log-softmax, compute log-LR.
Environment: `refs/ProteinGym/proteingym/baselines/evoscale/evoscale_env.yml`

**Two APIs:**
```python
# SDK API (ESMC class — sequential reference)
from esm.models.esmc import ESMC
model = ESMC.from_pretrained("esmc_600m").to("cuda")
protein_tensor = model.encode(ESMProtein(sequence=seq))
logits = model.logits(masked_tensor, LogitsConfig(sequence=True)).logits.sequence

# HuggingFace-style API (EsmcForMaskedLM — batched shinkaevolve)
from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
model = EsmcForMaskedLM.from_pretrained("biohub/ESMC-600M", dtype=torch.float16).cuda()
logits = model(input_ids=batch_ids, attention_mask=batch_mask).logits  # (B, L+2, 64)
```

**ESMC vs ESM-2:**
- Architecture: Pre-LN + RoPE + SwiGLU (no bias) vs Post-LN + learned pos emb + GELU
- Training: 2.8B sequences (UniRef90 + more) vs 250M sequences (UniRef50)
- Vocab: 64 tokens vs 33 tokens
- Model: new weights (not distilled from ESM-2)
- License: MIT (EvolutionaryScale) vs non-commercial (Meta)

---

## 6. Papers

| Paper | arXiv | What it contributes |
|---|---|---|
| Lin et al. 2023 (ESM-2) | https://arxiv.org/abs/2202.03036 | ESM-2 architecture, training, weights |
| Meier et al. 2021 (ESM-1v) | https://www.biorxiv.org/content/10.1101/2021.07.09.450648 | Scoring strategies: wt-marginals, masked-marginals, pseudo-ppl |
| Notin et al. 2023 (ProteinGym) | https://arxiv.org/abs/2303.01626 | Benchmark, 217 assays, ρ=0.414 baseline |
| Kantroo, Wagner & Machta 2024 (OFS) | https://arxiv.org/abs/2407.07265 | One Fell Swoop — 1-pass via trained MLP heads |

---

## 6. ESMC 300M — EvolutionaryScale (local / Mac MPS)

**Submodule:** `refs/evolutionaryscale-esm/` (same repo as ESMC 600M, v3.4.1+)
**HuggingFace hub:** https://huggingface.co/biohub/ESMC-300M
**SDK alias:** `esmc_300m`
**License:** MIT

Key differences vs ESMC 600M:
| | ESMC-300M | ESMC-600M |
|---|---|---|
| Params | 333M | 600M |
| Layers | 30 | 36 |
| Hidden dim | 960 | 1152 |
| Heads | 15 | 16 |
| Weight file | ~638MB bfloat16 | ~1.2GB bfloat16 |

**Local Mac inference:**
```bash
# one-time env setup
python3 -m venv .venv-esmc
.venv-esmc/bin/pip install "esm>=3.0.0"

# run
.venv-esmc/bin/python3 scoring/esmc/local_300m.py
```

Observed on M3 / 8GB:
- Load time: ~4s (weights cached at `~/.cache/huggingface/hub/`)
- Inference: ~300ms per sequence after Metal shader warm-up (first pass ~1.8s)
- Memory: ~700MB unified RAM
- dtype: bfloat16 (MPS) — matches GPU behaviour

**Flash Attention 2 note:** the `flash_attn` package is CUDA-only (no MPS backend).
`use_flash_attn=True` requests `attn_implementation="flash_attention_2"` from HuggingFace,
but without `flash_attn` installed HuggingFace silently falls back to SDPA.
Result: `use_flash_attn=True` and `False` behave identically on MPS.

Key files:
| File | What it does |
|---|---|
| `scoring/esmc/local_300m.py` | Local MPS inference script with demo (SNCA + GFP) |
| `scoring/esmc/masked_marginals.py` | Masked-marginals scoring (GPU/Modal) |
| `scoring/esmc/batched_masked.py` | Batched variant for throughput |

---

## 7. Known deviations (our code vs ProteinGym original)

| # | What | Impact | Affects |
|---|---|---|---|
| 1 | `transformers` vs `fair-esm` | Unknown — validated by `make gate` (eager≡sdpa, not transformers≡fair-esm) | All methods |
| 2 | `float16` vs `float32` | < 0.002 ρ expected | All methods |
| 3 | Window centre off by 1 char for L>1022 | < 0.001 ρ on long-sequence assays | masked_marginals, batched_masked |
| 4 | Skip BOS/EOS mask passes | None — those positions not used for scoring | masked_marginals |

Run `python check_consistency.py` to measure deviations 1 and 2 empirically.
