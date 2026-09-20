# ESM-2: What Are We Actually Doing, and What's the Highest-Impact $1000 Move?

*Niall O'Leary · September 2026*

---

## The Setup

We have ESM-2 650M. 33 transformer layers. 1280-dimensional residue embeddings. 2.6 GB of fp32 weights trained on 250M protein sequences (~65B tokens) from UniRef50. We have ProteinGym: 217 deep mutational scanning assays, 2.4M measured variant effects across 186 unique proteins. We have a live two-track benchmark reproducing Notin et al. NeurIPS 2023 — currently sitting at ρ=+0.4441 on wildtype marginals (published: 0.430), masked marginals converging toward 0.440.

We have $1000 and a deadline.

The question is: for some dataset D, weights W, and compute budget B — what is the most economically upstream intervention that meaningfully improves variant effect prediction?

---

## What ESM-2 Is Actually Computing

Before the strategy question, the framing question. What *is* ESM-2 doing when it scores a variant?

ESM-2 is a masked language model. Its training objective:

```
L = -E[ log p(x_i | x_{-i}) ]
```

Minimising this cross-entropy is equivalent to maximising the **mutual information** between each residue and its context. Over 65B tokens of evolutionary sequence, the model learns to approximate the conditional distribution P(x_i | x_{-i}) — the probability that amino acid a belongs at position i, given every other position in the protein.

The bidirectional attention is doing a learned soft query over the amino acid sequence manifold. Each of the 33 attention layers refines the per-residue representation h_i ∈ ℝ¹²⁸⁰ by aggregating weighted information from every other position. By layer 33, h_i is the **sufficient statistic for predicting x_i from x_{-i}** — a compressed encoding of all the evolutionary and structural constraints bearing on that residue. The final linear projection collapses this to a 33-dimensional probability simplex: the model's posterior over which amino acid belongs here.

The masked marginals score for a variant:

```
Score = Σ_i∈mutations [ log p(mut_i | x_{-i}) − log p(wt_i | x_{-i}) ]
      = Σ_i  log [ p(mut_i | context) / p(wt_i | context) ]
      = Σ_i  surprise(wt) − surprise(mut)
```

This is a **log-likelihood ratio** — the log Bayes factor comparing the mutant to wildtype under the model's learned distribution. Positive score means the mutation is *less surprising* to the model given evolutionary context. Negative score means the model has never seen this substitution favoured in 250M sequences of evolution.

Critically: this is not a physical free energy. It is an **information-theoretic plausibility ratio** under a learned evolutionary prior. The fact that it correlates with DMS fitness at ρ≈0.44 means evolutionary plausibility is a reasonable but imperfect proxy for molecular fitness — which is exactly what we'd expect.

---

## What the Benchmark Is Telling Us

Three numbers from our live run:

| Method | Our ρ | Published ρ | Gap to published |
|---|---|---|---|
| wt_marginals | +0.4441 | 0.430 | +0.014 above |
| masked_marginals | +0.4374 (88% done) | 0.440 | on target |
| pseudo_ppl | queued | 0.440 | — |

We're reproducing the paper. The model is doing what it says on the tin.

Now the uncomfortable number: **GPU MFU 7.8%** against A100 fp32 peak of 19.5 TFLOPS. We are utilising less than 8% of the hardware.

Why? The masked marginals algorithm requires L sequential forward passes per protein (one per unique mutation position). For a protein of length L=397 (average), that's 397 passes × ~53ms/pass = ~21 seconds/assay. This is not a compute-bound problem — it's a **memory-bandwidth-bound, serialised loop**. The GPU spends most of its time waiting for weights to move from HBM into compute units, one pass at a time, with no batching across mask positions.

This means: every speedup technique that addresses the serialisation — batching, kernel fusion, KV-cache-equivalent tricks — has outsized returns because we're so far from the hardware ceiling.

---

## The Four Moves and Their True Costs

### Move 1: Batched Multi-Mask Inference Kernel

The observation: across all L forward passes for a single protein, the K and V matrices for every **unmasked** position are identical. Only K[i] = K_mask changes per pass. This is a rank-1 substitution per pass over an otherwise fixed K/V base.

A Triton kernel that holds the shared K/V in SRAM and batches all L mask positions simultaneously is the ESM-specific equivalent of KV-cache prefix sharing in autoregressive models — the same insight vLLM uses, but adapted for masked LM inference. Neither vLLM nor SGLang implements this; they assume causal attention. The gap is open.

Expected speedup: 5–10× on L=400, more on longer sequences (O(L²) → O(L) amortised over positions).

Budget: ~$50 compute + 3–5 days dev. Downstream effect: every other experiment in this list gets 5–10× cheaper.

### Move 2: Test-Time Adaptation (TTA) via MSA Fine-Tuning

ESM-2 was trained on UniRef50 — a 30% sequence identity cluster. For most benchmark proteins, the model has never seen the *specific family* in detail. But homologous sequences for most ProteinGym proteins exist freely in UniRef90/UniRef100.

The play: for each protein before scoring, run 100–500 steps of masked LM fine-tuning on just its MSA (homologous sequences, no fitness labels). Adapt the model's prior to be family-specific. Then score variants.

This is unsupervised, requires no DMS labels, and exploits information that demonstrably exists (proteins with deep MSAs score better on all methods). The compute cost: ~$0.50–$2/protein × 217 proteins = $100–400 within budget.

Expected gain: +0.02–0.05 ρ. Gets us from 0.44 toward 0.47–0.49. Potentially beats methods like EVmutation and ESM-1v on average ρ.

Key question: does TTA on a held-out set (proteins not used during any adaptation) generalise? This is the experiment worth running.

### Move 3: Mechanistic Interpretability via SAE Features

InterPLM (Nature Methods 2025) trained sparse autoencoders on ESM-2's residual stream activations and found 2,548 interpretable features per layer — features that fire for "buried hydrophobic residue", "solvent-exposed loop", "active site", "coiled-coil". These are circuit-level descriptions of what the model computed.

The hypothesis: there exist identifiable circuits in ESM-2 that are causally responsible for the fitness signal in the masked marginals score. If we can identify and selectively amplify these circuits (or prune the rest), we improve the signal-to-noise ratio in variant scoring without changing the model's evolutionary prior.

The novel gap: InterPLM showed interpretable features exist. No paper has demonstrated a closed loop: find feature → intervene → measure ProteinGym ρ → publish. This is the open problem, and it is squarely in the 2027 novelty window.

Budget: mostly engineering time. Compute: <$100 to run feature correlation analysis across 217 assays. The hard part is the methodology, not the compute.

### Move 4: Continual Pre-Training on UniRef90

ESM-2 was trained on UniRef50 — ~250M sequences, ~65B tokens. A protein-specific Chinchilla scaling law (NeurIPS 2024) predicts optimal training for 650M parameters requires ~2× more tokens than ESM-2 received. The model is undertrained.

UniRef90 has ~250M more sequences and richer evolutionary diversity. Extending training for 50–100K steps:

```
50K steps × batch 512 × 1024 tokens = 26B additional tokens (~$840 at A100 $2.10/hr)
```

This fits within $1000. But expected ρ gain is uncertain and the feedback loop is slow (train → benchmark → observe, 3+ days). It's the highest-variance option.

---

## The $1000 Decision

Here is the honest ranking by expected impact per dollar:

| Intervention | Expected Δρ | Compute Cost | Dev Time | Novelty |
|---|---|---|---|---|
| Batched multi-mask kernel | 0 (enables others) | ~$50 | 3–5 days | Medium |
| TTA / MSA fine-tuning | +0.02–0.05 | ~$200–400 | 3–5 days | Medium-High |
| SAE mech-interp circuit analysis | +0.03–0.08 (uncertain) | ~$100 | 2–4 weeks | High |
| Continual pre-training | +0.01–0.03 | ~$840 | 1–2 days | Low |

**Recommendation: kernel first, then TTA.**

The inference kernel is the enabling move. It is not an optimisation — it is a qualitative change in the feedback loop. Once each iteration costs $10 instead of $50, every other experiment becomes viable. The kernel also has its own publication story: a benchmarked speedup on ESM-2 inference with ProteinGym as the correctness harness, with the novel batched-multi-mask implementation as the contribution.

TTA is the highest expected-ρ move within budget. The experiment is clean: run baseline masked marginals → run TTA-adapted masked marginals → measure per-assay ρ change → correlate with MSA depth. This is a paper.

The mech-interp angle is the highest-novelty, longest-timeline play. It needs the benchmark infrastructure we're building now. It is the 2027 story, not the 2026 sprint.

---

## The Comparable Ground Truth Question

The DMS fitness scores *are* our ground truth. There is no better one.

ProteinGym covers 217 assays across 186 proteins, 2.4M measured variant effects from systematic mutagenesis → selection → Illumina sequencing. This is not a proxy — it is the direct measurement of what the model's score is supposed to predict.

The gap between ρ=0.44 and ρ=1.0 has three components:

1. **Model capacity** — undertrained by ~2×, attentional circuits that don't fully capture long-range contacts
2. **Proxy mismatch** — evolutionary plausibility ≠ DMS fitness. Evolution optimises reproductive success; DMS measures a specific biochemical phenotype in a specific assay condition
3. **Information ceiling** — some fitness variance is irreducible from sequence alone (expression context, solvent, temperature, binding partner)

Component 3 puts a hard ceiling on sequence-only models, estimated at ρ≈0.6–0.7 for most assays. We are at 0.44. The gap to the ceiling is real and addressable. The ceiling itself is not.

---

## What We Are Actually Doing

The most upstream economic description of this project:

We are building the cheapest, most reproducible pipeline to evaluate interventions on ESM-2's evolutionary prior against a ground-truth fitness benchmark. The benchmark infrastructure is the asset. Every intervention — kernel, TTA, mech-interp, pre-training — is a bet placed against it.

The inference speedup reduces the cost per bet. TTA improves the quality of the prior. The mech-interp work explains which parts of the prior matter. The pre-training extends the prior itself.

None of this requires building a new model. The 650M weights trained on 65B evolutionary tokens are a standing wave of evolutionary information, and we are learning how to query them better.

---

## Appendix: Key Numbers

```
ESM-2 650M:         650M params, 33L × d1280 × h20 × ffn5120
Training data:      UniRef50 2021_04, ~250M seqs, ~65B tokens
Chinchilla gap:     ~2× undertrained (protein-specific D∝N^1.30, NeurIPS 2024)
Model size:         2.6 GB fp32 / 1.3 GB fp16
HF downloads:       ~1.58M/month (top 50 scientific models)

Benchmark:          ProteinGym 217 DMS substitution assays, 2.4M variants
Our results:        wt=0.4441, mm=0.4374 (88% complete), ppl=queued
Published:          wt=0.430, mm=0.440, ppl=0.440

Compute:            A100-SXM4-40GB @ $2.10/hr
MFU:                7.8% (memory-bandwidth bound, sequential loop)
O(L²) constant:     β = 1.337×10⁻⁴ s/AA², confirmed empirically
Avg assay time:     ~21s (L=397)

TTA budget:         ~$200–400 for all 217 proteins
Kernel dev:         ~$50 compute + 3–5 days
SAE features:       2,548/layer (InterPLM, Nature Methods 2025)
INT8 quant risk:    one assay 0.591→0.223 (Aug 2026 bioRxiv) — use per-assay AUC guard
```
