"""
Track C MFU research ideas — three optimization approaches on ESMC 600M.

All use EsmcForMaskedLM with true-batch scoring (B sequences per forward pass,
each with ONE position masked — identical scores to sequential, exact, no approximation).

The model is loaded EXTERNALLY (in the Modal function) with the appropriate config.
These wrappers select the batch size and delegate to batched_masked.score_variants.

Research ideas:

  IDEA 1 — torch.compile reduce-overhead (B=32)
    Model: EsmcForMaskedLM + torch.compile(mode="reduce-overhead")
    Target: ~20-40% MFU improvement from reduced Python dispatch overhead
    Cost: ~30-60s one-time compilation warmup

  IDEA 2 — torch.compile max-autotune (B=64)
    Model: EsmcForMaskedLM + torch.compile(mode="max-autotune")
    Target: ~40-60% MFU improvement from Triton kernel autotuning + larger batch
    Cost: ~2-5 min one-time compilation

  IDEA 3 — Flash Attention 2 + compile + B=32
    Model: EsmcForMaskedLM(attn_implementation="flash_attention_2") + torch.compile
    Target: 2-4× attention speedup, especially for long sequences (L>500)
    Cost: requires flash-attn package; 2-4× faster attention computation

All three maintain exact Spearman ρ identical to batched_masked (no approximation).
"""

from scoring.esmc.batched_masked import score_variants as _batched_score


def score_variants_idea1(model, tokenizer, sequence, variants, device="cuda"):
    """Idea 1: torch.compile reduce-overhead, B=32."""
    return _batched_score(model, tokenizer, sequence, variants, device, batch_size=32)


def score_variants_idea2(model, tokenizer, sequence, variants, device="cuda"):
    """Idea 2: torch.compile max-autotune, B=64."""
    return _batched_score(model, tokenizer, sequence, variants, device, batch_size=64)


def score_variants_idea3(model, tokenizer, sequence, variants, device="cuda"):
    """Idea 3: Flash Attention 2 + compile, B=32."""
    return _batched_score(model, tokenizer, sequence, variants, device, batch_size=32)
