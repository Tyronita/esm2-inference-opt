# Known Limitations

## Per-method limitations

### wt_marginals
- **Self-information leakage**: model sees position i when scoring it. The output distribution at position i includes information from x_i itself, so the score is less position-isolated than masked_marginals. Systematically produces overconfident (sharper) distributions.
- **No positional isolation**: correlated positions (e.g. disulfide partners) contaminate each other's scores.
- Expected ρ ≈ 0.40 (below masked_marginals).

### masked_marginals
- **O(L) forward passes**: SNCA (L=140) = 140 passes, LRRK2 (L=2527) = 2527 passes. Full ProteinGym run on A100 ≈ 2 hours.
- **Windowing for L > 1022**: ESM-2 max input is 1024 tokens (incl. BOS/EOS). Sequences longer than 1022 AA use a 1024-token window centred on the mutation site. Positions near window boundaries see truncated context.
- **Cannot batch across variants** without masking multiple positions simultaneously.

### pseudo_ppl
- Same O(L) cost as masked_marginals.
- For single-site variants (the majority in ProteinGym): **identical to masked_marginals**. Contexts are the same because the one mutated position is masked.
- Meaningful difference only for multi-site variants; ProteinGym single-substitution benchmark makes this method redundant.

### batched_masked (B=8, B=32)
- **Approximation error from multi-masking**: when B positions are masked simultaneously, each masked token sees the other B-1 masked tokens as `[MASK]` in its attention context. With single-position masking (masked_marginals) it sees the true wildtype residue.
- Error grows with B and with sequence length (more positions affected by each mask).
- For small proteins (L < 100) and small B (≤8), error is typically < 0.01 ρ.
- Not validated for sequences with strong long-range dependencies.

### simple_ofs
- **NOT the Kantroo et al. 2024 OFS method.**
- The true OFS uses an ensemble of 8 trained MLPs (one per residue position type) that map unmasked ESM-2 hidden states → predicted masked distributions. These MLPs are trained specifically on ESM-2 650M.
- This implementation is functionally equivalent to `wt_marginals` (same forward pass, same logits).
- Expected to match wt_marginals performance (ρ ≈ 0.40), not masked_marginals (ρ ≈ 0.44).
- Reference: Kantroo, Wagner & Machta (2024), PRX Life, arXiv:2407.07265.

## System-level limitations

### Sequence length cap
ESM-2 650M was trained with `max_position_embeddings = 1026` (1024 residues + BOS + EOS). Any sequence longer than 1022 AA requires windowing. ProteinGym contains proteins up to ~2500 AA (LRRK2 = 2527 AA); windowing introduces boundary effects.

### Vocabulary
ESM-2's tokenizer covers the 20 standard amino acids plus special tokens. Non-canonical residues (selenocysteine, pyrrolysine, modified AAs) are not in the vocabulary and will tokenize as `<unk>`.

### No structural information
All methods here are sequence-only. Structure-aware models (SaProt, ProtSSN) consistently outperform sequence-only ESM-2 on ProteinGym (Spearman ρ ≈ 0.49–0.52 vs 0.44). The structure bottleneck is not addressed by any scoring method here.

### No MSA / evolutionary context
Single-sequence only. MSA Transformer and TranceptEVE use multiple sequence alignments and consistently outperform single-sequence ESM-2. Retrieval-augmented methods (TranceptEVE L) achieve ρ ≈ 0.56.

### fp16 precision
All models loaded as `torch_dtype=float16` for memory efficiency. For borderline variants (score near 0), fp16 rounding can flip sign predictions. fp32 scoring would increase precision but doubles memory usage (~2.6GB vs 1.3GB).

### Metric limitations
- **Spearman ρ** measures rank correlation — insensitive to score magnitude.
- **NDCG** requires non-negative relevance; fitness is shifted to [0, ∞) which changes scale-sensitivity.
- **AUC** only available for assays with binary labels (`DMS_score_bin`); not all 217 assays have these.
- **Fraction correct** is sensitive to the median split threshold; aggressive DMS distributions can make this misleading.

## What this repo does NOT implement
- True OFS (Kantroo et al. 2024) — requires trained MLP heads
- MSA-augmented scoring (requires MSA retrieval pipeline)
- Structure-conditioned scoring (requires AlphaFold2/ESMFold structure prediction)
- Indel scoring (masked_marginals is substitution-only; OFS handles indels)
- ESM-2 3B / 15B (larger models may improve ρ but require >40GB GPU)
