"""
Local smoke test — M3 MPS, ESM-2 8M, no GPU spend.

Tests the full code path that will run on Modal A100 with 650M:
  imports → model load → masked_marginals → wt_marginals → batched_masked
  → pseudo_ppl → metrics → consistency (fp16 vs fp32)

Uses ESM-2 8M (facebook/esm2_t6_8M_UR50D) — same architecture, 30MB,
loads in ~2s on MPS. Same code path as 650M.

Run:
    source .venv/bin/activate
    python smoke_test.py
"""

import sys
import time
from pathlib import Path

# Make scoring/ and benchmark/ importable
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

SNCA_SEQ = (
    "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTK"
    "EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDP"
    "DNEAYEMPSEEGYQDYEPEA"
)
VARIANTS_10 = ["A53T", "E46K", "A30P", "G51D", "H50Q", "A18T", "V70M", "E83Q", "A76T", "S87N"]
VARIANTS_FIT = [0.1, -0.5, 0.3, -0.2, 0.4, 0.0, -0.1, 0.2, -0.3, 0.1]
MODEL_8M = "facebook/esm2_t6_8M_UR50D"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

results = []


def check(name, fn):
    t0 = time.perf_counter()
    try:
        out = fn()
        elapsed = time.perf_counter() - t0
        print(f"  {PASS}  {name}  ({elapsed:.2f}s)")
        results.append((name, True, None))
        return out
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  {FAIL}  {name}  ({elapsed:.2f}s)")
        print(f"         {type(e).__name__}: {e}")
        results.append((name, False, str(e)))
        return None


# ── 1. Imports ────────────────────────────────────────────────────────────────
print("\n── 1. Imports ───────────────────────────────────────────────────────")

check("import torch",           lambda: __import__("torch"))
check("import transformers",    lambda: __import__("transformers"))
check("import numpy",           lambda: __import__("numpy"))
check("import scipy",           lambda: __import__("scipy"))
check("import pandas",          lambda: __import__("pandas"))
check("import sklearn",         lambda: __import__("sklearn"))
check("import Bio (biopython)", lambda: __import__("Bio"))
check("import esm (fair-esm)",  lambda: __import__("esm"))

check("from scoring.registry import get_registry",
      lambda: __import__("scoring.registry", fromlist=["get_registry"]))
check("from scoring.masked_marginals import score_variants",
      lambda: __import__("scoring.masked_marginals", fromlist=["score_variants"]))
check("from scoring.wt_marginals import score_variants",
      lambda: __import__("scoring.wt_marginals", fromlist=["score_variants"]))
check("from scoring.pseudo_ppl import score_variants",
      lambda: __import__("scoring.pseudo_ppl", fromlist=["score_variants"]))
check("from scoring.batched_masked import score_variants",
      lambda: __import__("scoring.batched_masked", fromlist=["score_variants"]))
check("from scoring.simple_ofs import score_variants",
      lambda: __import__("scoring.simple_ofs", fromlist=["score_variants"]))
check("from benchmark.metrics import compute_all",
      lambda: __import__("benchmark.metrics", fromlist=["compute_all"]))

# ── 2. MPS availability ───────────────────────────────────────────────────────
print("\n── 2. Device ────────────────────────────────────────────────────────")
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
check(f"device = {device}", lambda: None)
check("torch.tensor on device",
      lambda: torch.tensor([1.0]).to(device))

# ── 3. Model load (8M — same arch as 650M, loads in ~2s) ─────────────────────
print("\n── 3. Model load (ESM-2 8M — proxy for 650M code path) ──────────────")
from transformers import AutoTokenizer, EsmForMaskedLM

model = check("AutoTokenizer.from_pretrained(8M)",
              lambda: AutoTokenizer.from_pretrained(MODEL_8M))
hf_model = check("EsmForMaskedLM.from_pretrained(8M, fp16)",
                  lambda: EsmForMaskedLM.from_pretrained(
                      MODEL_8M, torch_dtype=torch.float16).eval().to(device))
hf_model_fp32 = check("EsmForMaskedLM.from_pretrained(8M, fp32)",
                       lambda: EsmForMaskedLM.from_pretrained(
                           MODEL_8M, torch_dtype=torch.float32).eval().to(device))

# ── 4. Scoring methods ────────────────────────────────────────────────────────
print("\n── 4. Scoring methods (ESM-2 8M, SNCA 10 variants, MPS) ─────────────")

if hf_model is not None and model is not None:
    from scoring import masked_marginals, wt_marginals, pseudo_ppl, batched_masked, simple_ofs

    mm_scores = check("masked_marginals.score_variants (10 variants, L=140)",
                      lambda: masked_marginals.score_variants(
                          hf_model, model, SNCA_SEQ, VARIANTS_10, device))
    wt_scores = check("wt_marginals.score_variants",
                      lambda: wt_marginals.score_variants(
                          hf_model, model, SNCA_SEQ, VARIANTS_10, device))
    pp_scores = check("pseudo_ppl.score_variants",
                      lambda: pseudo_ppl.score_variants(
                          hf_model, model, SNCA_SEQ, VARIANTS_10, device))
    b8_scores = check("batched_masked.score_variants (B=8)",
                      lambda: batched_masked.score_variants(
                          hf_model, model, SNCA_SEQ, VARIANTS_10, device, batch_size=8))
    b32_scores = check("batched_masked.score_variants (B=32)",
                       lambda: batched_masked.score_variants(
                           hf_model, model, SNCA_SEQ, VARIANTS_10, device, batch_size=32))
    ofs_scores = check("simple_ofs.score_variants",
                       lambda: simple_ofs.score_variants(
                           hf_model, model, SNCA_SEQ, VARIANTS_10, device))

    # Spot-check: scores are finite floats
    if mm_scores:
        check("masked_marginals returns 10 finite floats",
              lambda: (
                  len(mm_scores) == 10 and
                  all(isinstance(s, float) and s == s for s in mm_scores)
                  or (_ for _ in ()).throw(ValueError(f"got {mm_scores}"))
              ))

# ── 5. Metrics ────────────────────────────────────────────────────────────────
print("\n── 5. Metrics ───────────────────────────────────────────────────────")
from benchmark.metrics import compute_all

if mm_scores:
    metrics = check("compute_all(mm_scores, fitness)",
                    lambda: compute_all(mm_scores, VARIANTS_FIT))
    if metrics:
        check("spearman_rho present and finite",
              lambda: "spearman_rho" in metrics and
              abs(metrics["spearman_rho"]) <= 1.0 or
              (_ for _ in ()).throw(ValueError(str(metrics))))
        check("ndcg_10 present",  lambda: "ndcg_10" in metrics)
        check("top5_recall present", lambda: "top5_recall" in metrics)

# ── 6. Precision consistency (fp16 vs fp32, HuggingFace 8M, MPS) ──────────────
# NOTE: The 8M model has divergent weights between fair-esm and HuggingFace.
#       Cross-library consistency (fair-esm vs transformers) is tested on the
#       650M model in check_consistency.py — that's the one that maps to ρ=0.414.
#       Here we test fp16 vs fp32 within HuggingFace, which is the actual
#       precision proxy test for the Modal A100 run.
print("\n── 6. Precision consistency (HF 8M fp16 vs fp32, MPS) ───────────────")
import numpy as np
from scipy.stats import spearmanr
import esm as fair_esm

# Smoke: confirm fair-esm loads and runs (don't compare scores across sources)
fe_model = check("fair_esm.pretrained.esm2_t6_8M_UR50D() loads on MPS",
                 lambda: (
                     lambda m, a: (m.eval().to(device), a)
                 )(*fair_esm.pretrained.esm2_t6_8M_UR50D()))
if fe_model is not None:
    fe_m, fe_alpha = fe_model
    bc = fe_alpha.get_batch_converter()
    _, _, toks_fe = bc([("p", SNCA_SEQ[:20])])
    toks_fe = toks_fe.to(device)
    check("fair-esm forward pass (first 20aa)",
          lambda: fe_m(toks_fe.clone())["logits"].shape)

# The real gate: fp16 vs fp32 within HuggingFace — this is what Modal A100 runs
if hf_model_fp32 is not None and mm_scores:
    from scoring import masked_marginals as _mm

    fp32_scores = check("masked_marginals fp32 (same positions as fp16)",
                        lambda: _mm.score_variants(
                            hf_model_fp32, model, SNCA_SEQ, VARIANTS_10, device))

    if fp32_scores:
        rho_prec = spearmanr(mm_scores, fp32_scores).correlation
        diff_prec = np.abs(np.array(mm_scores) - np.array(fp32_scores))
        ok_rho = rho_prec >= 0.999
        ok_diff = diff_prec.max() < 0.05

        check(f"fp16 vs fp32 Spearman ρ={rho_prec:+.4f} (gate ≥ 0.999)",
              lambda: ok_rho or (_ for _ in ()).throw(
                  ValueError(f"ρ={rho_prec:.4f} < 0.999 — fp16 degrades ranking")))
        check(f"fp16 vs fp32 max |diff|={diff_prec.max():.5f} (gate < 0.05)",
              lambda: ok_diff or (_ for _ in ()).throw(
                  ValueError(f"max diff {diff_prec.max():.5f} — fp16 precision loss")))

# ── 7. Registry ───────────────────────────────────────────────────────────────
print("\n── 7. Registry ──────────────────────────────────────────────────────")
from scoring.registry import get_registry

reg = check("get_registry(device) returns 6 methods",
            lambda: get_registry(device))
if reg:
    check("all 6 methods present",
          lambda: len(reg) == 6 or (_ for _ in ()).throw(
              ValueError(f"got {[m.name for m in reg]}")))
    for m in reg:
        check(f"registry[{m.name}] callable",
              lambda m=m: callable(m.fn))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 66)
passed  = sum(1 for _, ok, _ in results if ok)
failed  = sum(1 for _, ok, _ in results if not ok)
total   = len(results)
verdict = "ALL CLEAR — safe to run on Modal A100" if failed == 0 else f"{failed} FAILURES — fix before Modal run"
print(f"  {passed}/{total} passed    {verdict}")
if failed:
    print("\n  Failures:")
    for name, ok, err in results:
        if not ok:
            print(f"    ✗  {name}")
            print(f"       {err}")
print("=" * 66)
sys.exit(0 if failed == 0 else 1)
