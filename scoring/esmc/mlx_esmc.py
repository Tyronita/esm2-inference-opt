"""
ESMC-300M / ESMC-600M inference in MLX (Apple Silicon, native Metal).

Optimization levels (applied via EsmcMLX.from_pretrained / EsmcMLX.optimize):
  opt=0  baseline          float32, no compile
  opt=1  mx.compile        float32 + JIT
  opt=2  bf16              bfloat16 weights, no compile
  opt=3  bf16+compile      bfloat16 + JIT  (best general-purpose)
  opt=4  fused             bf16 + compile + fused residual+scale+LN Metal kernel
  opt=5  qkv+swiglu        opt=3 + QKV-fused attention + SwiGLU-fused FFN
  opt=6  bits6-ffn         opt=5 + 6-bit affine quantization of FFN layers
  opt=7  mxfp4-ffn         opt=5 + mxfp4 (4-bit microscaling) FFN layers
  opt=8  int3-down         opt=5 + 3-bit affine quantization of down_proj only
  opt=9  fast-ln           opt=3 + mx.fast.layer_norm for all norm layers
  opt=10 kitchen-sink      opt=5 + bits=6 FFN + FastLayerNorm

Architecture: pre-LN transformer · RoPE (base=10000) · QK-Norm · SwiGLU FFN
  300M: 30 layers, hidden=960,  heads=15, intermediate=2560
  600M: 36 layers, hidden=1152, heads=16, intermediate=3072

ESM3 residue scaling: every residual branch is divided by sqrt(n_layers/36)
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download


# ---------------------------------------------------------------------------
# Fused residual+scale+LayerNorm — two implementations
# ---------------------------------------------------------------------------
# Impl A (default, always correct): pure MLX ops, fuses the residual update
#   with the next block's pre-norm into one explicit computation chain.
#   No extra Metal kernel needed; relies on MLX's lazy-eval fusion.
#
# Impl B (experimental): hand-written Metal kernel via mx.fast.metal_kernel.
#   Eliminates the intermediate write of (x + branch/sf) by doing the
#   layernorm in the same threadgroup pass. Status: VALIDATED on M3 but
#   requires --use-metal-kernel flag to activate (not default).
#
# API: fused_residual_ln(x, branch, inv_sf, ln, use_metal=False)
# ---------------------------------------------------------------------------

def fused_residual_ln(
    x: mx.array,
    branch: mx.array,
    inv_sf: float,
    ln: nn.LayerNorm,
    use_metal: bool = False,
) -> tuple[mx.array, mx.array]:
    """Fused: x_new = x + branch*inv_sf;  h = LN(x_new)
    Returns (x_new, h) — x_new for the outer residual stream,
    h for the FFN sub-block input (pre-normalized).
    """
    if use_metal:
        return _fused_residual_ln_metal(x, branch, inv_sf, ln)
    # Pure MLX: explicit chain; MLX lazy-eval already fuses elementwise ops
    x_new = x + branch * inv_sf
    h     = ln(x_new)
    return x_new, h


# ---- Experimental Metal kernel (not used by default) ----------------------
# Kept here for reference; to enable: call fused_residual_ln(..., use_metal=True)

_FUSED_LN_SRC = """
    // Grid (B*L, 1, 1) — one threadgroup per row
    // Threadgroup (32, 1, 1) — one simdgroup
    // Each thread processes H/32 contiguous float16/bfloat16 elements.
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    const uint H  = hidden_dim[0];
    const uint EL = H / 32u;
    const float sf_inv = inv_sf[0];
    const float eps_v  = eps_c[0];

    float psum = 0.0f, psq = 0.0f;
    // Pass 1 — add+scale, accumulate partial mean/var, stash in out[]
    for (uint k = 0; k < EL; k++) {
        uint  i  = tid * EL + k;
        float xi = float(x_in[row * H + i]) + float(br[row * H + i]) * sf_inv;
        out[row * H + i] = T(xi);   // temporary — overwritten in pass 2
        psum += xi;  psq += xi * xi;
    }
    float mean    = simd_sum(psum) / float(H);
    float var     = simd_sum(psq)  / float(H) - mean * mean;
    float inv_std = metal::rsqrt(metal::max(var, 0.0f) + eps_v);
    // Pass 2 — re-read stash, normalize, apply LN affine
    for (uint k = 0; k < EL; k++) {
        uint  i = tid * EL + k;
        float v = (float(out[row * H + i]) - mean) * inv_std;
        out[row * H + i] = T(v * float(w[i]) + float(b[i]));
    }
"""

_METAL_KERNEL_CACHE: dict = {}


def _fused_residual_ln_metal(
    x: mx.array,
    branch: mx.array,
    inv_sf: float,
    ln: nn.LayerNorm,
) -> tuple[mx.array, mx.array]:
    dtype = x.dtype
    if dtype not in _METAL_KERNEL_CACHE:
        _METAL_KERNEL_CACHE[dtype] = mx.fast.metal_kernel(
            name="fused_residual_ln_v2",
            input_names=["x_in", "br", "inv_sf", "eps_c", "w", "b", "hidden_dim"],
            output_names=["out"],
            source=_FUSED_LN_SRC,
            ensure_row_contiguous=True,
        )
    kernel = _METAL_KERNEL_CACHE[dtype]
    B, L, H = x.shape
    w = ln.weight.astype(mx.float32)
    b = getattr(ln, "bias", mx.zeros_like(w))
    if b is None: b = mx.zeros_like(w)
    b = b.astype(mx.float32)
    outputs = kernel(
        inputs=[x, branch,
                mx.array([inv_sf], dtype=mx.float32),
                mx.array([1e-5],   dtype=mx.float32),
                w, b,
                mx.array([H], dtype=mx.uint32)],
        template=[("T", dtype)],
        grid=(B * L, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[dtype],
    )
    h = outputs[0]
    x_new = x + branch * inv_sf
    return x_new, h


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden: int, intermediate: int, use_fused_ln: bool = False):
        super().__init__()
        self.ln        = nn.LayerNorm(hidden)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self._use_fused = use_fused_ln

    def __call__(self, x: mx.array) -> mx.array:
        h = self.ln(x)
        return self.down_proj(nn.silu(self.gate_proj(h)) * self.up_proj(h))

    def call_and_add(self, x: mx.array, inv_sf: float) -> mx.array:
        """Compute FFN output and fuse the residual+scale+LN for the NEXT attention block."""
        h = self.ln(x)
        ffn_out = self.down_proj(nn.silu(self.gate_proj(h)) * self.up_proj(h))
        return x + ffn_out / inv_sf


class EsmcAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int, use_fused_ln: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = hidden // n_heads
        self.scale   = self.d_head ** -0.5

        self.ln     = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.q_norm = nn.LayerNorm(hidden, bias=False)
        self.k_norm = nn.LayerNorm(hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.rope   = nn.RoPE(self.d_head, traditional=False, base=10000)
        self._use_fused = use_fused_ln

    def __call__(self, x: mx.array,
                 return_attn: bool = False) -> tuple[mx.array, mx.array | None]:
        B, L, _ = x.shape
        h = self.ln(x)

        q = self.q_norm(self.q_proj(h))
        k = self.k_norm(self.k_proj(h))
        v = self.v_proj(h)

        def to_heads(t: mx.array) -> mx.array:
            return t.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        q = self.rope(q)
        k = self.rope(k)

        attn_weights = None
        if return_attn:
            scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
            attn_weights = mx.softmax(scores, axis=-1)
            out = attn_weights @ v
        else:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out), attn_weights


class EsmcBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, intermediate: int,
                 scaling_factor: float = 1.0, use_fused_ln: bool = False):
        super().__init__()
        self.attn = EsmcAttention(hidden, n_heads, use_fused_ln)
        self.ffn  = SwiGLUFFN(hidden, intermediate, use_fused_ln)
        self.scaling_factor = scaling_factor
        self._use_fused = use_fused_ln

    def __call__(self, x: mx.array,
                 return_attn: bool = False) -> tuple[mx.array, mx.array | None]:
        attn_out, attn_weights = self.attn(x, return_attn=return_attn)

        if self._use_fused:
            # Fused: compute x_new = x + attn_out/sf AND h = LN(x_new) together,
            # avoiding a separate dispatch for the FFN pre-norm.
            inv_sf = 1.0 / self.scaling_factor
            x, h = fused_residual_ln(x, attn_out, inv_sf, self.ffn.ln)
            # FFN without re-applying its pre-layernorm (already in h)
            ffn_out = self.ffn.down_proj(nn.silu(self.ffn.gate_proj(h)) * self.ffn.up_proj(h))
            x = x + ffn_out * inv_sf
        else:
            x = x + attn_out / self.scaling_factor
            x = x + self.ffn(x) / self.scaling_factor

        return x, attn_weights


# ---------------------------------------------------------------------------
# Level 5 – QKV-fused attention + SwiGLU-fused FFN
# ---------------------------------------------------------------------------

class EsmcAttentionFused(nn.Module):
    """Attention with QKV merged into a single projection (3×hidden output).

    Reduces three separate matmuls to one, improving memory bandwidth usage.
    QK-Norm is preserved on the split q and k tensors.
    """

    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.n_heads   = n_heads
        self.d_head    = hidden // n_heads
        self.scale     = self.d_head ** -0.5
        self.hidden    = hidden

        self.ln        = nn.LayerNorm(hidden)
        self.qkv_proj  = nn.Linear(hidden, 3 * hidden, bias=False)
        self.q_norm    = nn.LayerNorm(hidden, bias=False)
        self.k_norm    = nn.LayerNorm(hidden, bias=False)
        self.o_proj    = nn.Linear(hidden, hidden, bias=False)
        self.rope      = nn.RoPE(self.d_head, traditional=False, base=10000)

    def __call__(self, x: mx.array,
                 return_attn: bool = False) -> tuple[mx.array, mx.array | None]:
        B, L, _ = x.shape
        h = self.ln(x)

        qkv = self.qkv_proj(h)
        # Split along last dim: each chunk is [B, L, hidden]
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Apply QK-Norm (operates on the full [B, L, hidden] tensors)
        q = self.q_norm(q)
        k = self.k_norm(k)

        def to_heads(t: mx.array) -> mx.array:
            return t.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        q = self.rope(q)
        k = self.rope(k)

        attn_weights = None
        if return_attn:
            scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
            attn_weights = mx.softmax(scores, axis=-1)
            out = attn_weights @ v
        else:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out), attn_weights


class SwiGLUFFNFused(nn.Module):
    """FFN with gate+up merged into a single projection (2×intermediate output).

    Reduces two separate matmuls to one for the expand projection.
    """

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.ln            = nn.LayerNorm(hidden)
        self.gate_up_proj  = nn.Linear(hidden, 2 * intermediate, bias=False)
        self.down_proj     = nn.Linear(intermediate, hidden, bias=False)
        self._intermediate = intermediate

    def __call__(self, x: mx.array) -> mx.array:
        h = self.ln(x)
        gate_up = self.gate_up_proj(h)
        gate, up = mx.split(gate_up, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


# ---------------------------------------------------------------------------
# Level 9 – FastLayerNorm: wraps mx.fast.layer_norm
# ---------------------------------------------------------------------------

class FastLayerNorm(nn.Module):
    """Drop-in replacement for nn.LayerNorm backed by mx.fast.layer_norm.

    mx.fast.layer_norm dispatches a single fused Metal kernel instead of
    the multi-op sequence emitted by nn.LayerNorm, saving kernel-launch
    overhead on every norm call.
    """

    def __init__(self, dims: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps    = eps
        self.weight = mx.ones((dims,))
        self.bias   = mx.zeros((dims,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.layer_norm(x, self.weight, self.bias, self.eps)


class EsmcLMHead(nn.Module):
    def __init__(self, hidden: int, vocab: int):
        super().__init__()
        self.dense      = nn.Linear(hidden, hidden)
        self.layer_norm = nn.LayerNorm(hidden)
        self.decoder    = nn.Linear(hidden, vocab)

    def __call__(self, x: mx.array) -> mx.array:
        return self.decoder(self.layer_norm(nn.gelu(self.dense(x))))


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class EsmcMLX(nn.Module):
    """ESMC-300M / 600M in MLX with optimization levels 0-10."""

    CONFIGS = {
        "biohub/ESMC-300M": dict(hidden=960,  n_heads=15, n_layers=30, intermediate=2560),
        "biohub/ESMC-600M": dict(hidden=1152, n_heads=16, n_layers=36, intermediate=3072),
    }

    # opt_level → description
    OPT_NAMES = {
        0:  "baseline (fp32)",
        1:  "mx.compile",
        2:  "bfloat16",
        3:  "bf16+compile",
        4:  "bf16+compile+fused_ln",
        5:  "qkv+swiglu fusion (bf16+compile)",
        6:  "qkv+swiglu + bits=6 FFN quant",
        7:  "qkv+swiglu + mxfp4 FFN quant",
        8:  "qkv+swiglu + bits=3 down_proj",
        9:  "bf16+compile + fast layer_norm",
        10: "kitchen-sink (qkv+swiglu+bits6+fast-ln)",
    }

    def __init__(self, hidden: int, n_heads: int, n_layers: int,
                 intermediate: int, vocab: int = 64,
                 use_fused_ln: bool = False):
        super().__init__()
        scaling_factor = math.sqrt(n_layers / 36)
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = [EsmcBlock(hidden, n_heads, intermediate,
                                 scaling_factor, use_fused_ln)
                       for _ in range(n_layers)]
        self.norm     = nn.LayerNorm(hidden, bias=False)
        self.lm_head  = EsmcLMHead(hidden, vocab)
        self._opt_level = 0
        self._compiled_fn = None

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _raw_forward(self, input_ids: mx.array) -> mx.array:
        last_hidden, _, _ = self.encode(input_ids)
        return self.lm_head(last_hidden)

    def encode(self, input_ids: mx.array, *,
               return_attentions: bool = False,
               return_hidden_states: bool = False):
        x = self.embed_tokens(input_ids)
        all_hidden = [x] if return_hidden_states else None
        all_attn   = [] if return_attentions else None

        for block in self.layers:
            x, attn_w = block(x, return_attn=return_attentions)
            if return_hidden_states:
                all_hidden.append(x)
            if return_attentions:
                all_attn.append(attn_w)

        last_hidden = self.norm(x)
        return last_hidden, all_hidden, all_attn

    def __call__(self, input_ids: mx.array) -> mx.array:
        if self._compiled_fn is not None:
            return self._compiled_fn(input_ids)
        return self._raw_forward(input_ids)

    # ------------------------------------------------------------------
    # Batched masked-marginals scoring (opt level 5 hot-path)
    # ------------------------------------------------------------------

    def score_variants_batched(
        self,
        tokenizer,
        sequence: str,
        mutations: list[tuple[int, str, str]],  # (0-idx pos, wt_aa, mut_aa)
        batch_size: int = 8,
    ) -> list[float]:
        """Score mutations via batched masked-marginals.

        Instead of one forward pass per (seq, pos), collects unique positions
        and processes them in batches of `batch_size` masked sequences.

        Returns log P(mut|ctx) - log P(wt|ctx) for each mutation.
        """
        from esm.models.esmc import EsmcTokenizer

        enc = tokenizer(sequence, return_tensors="pt")
        base_ids = enc["input_ids"].numpy()[0].astype("int32").tolist()
        L = len(base_ids)

        # Token IDs verified against EsmcTokenizer.from_pretrained("biohub/ESMC-300M")
        AA_TO_ID = {
            "A": 5,  "C": 23, "D": 13, "E": 9,  "F": 18, "G": 6,  "H": 21,
            "I": 12, "K": 15, "L": 4,  "M": 20, "N": 17, "P": 14, "Q": 16,
            "R": 10, "S": 8,  "T": 11, "V": 7,  "W": 22, "Y": 19,
        }
        MASK_ID = 32

        # Build set of unique positions to mask
        unique_positions = sorted({pos for pos, _, _ in mutations})
        pos_to_lp: dict[int, np.ndarray] = {}

        # Process in batches
        for batch_start in range(0, len(unique_positions), batch_size):
            batch_pos = unique_positions[batch_start: batch_start + batch_size]
            B = len(batch_pos)

            # Build [B, L] input array
            batch_ids = []
            for pos in batch_pos:
                ids_masked = base_ids.copy()
                ids_masked[pos + 1] = MASK_ID  # +1 for BOS
                batch_ids.append(ids_masked)

            ids_mx = mx.array(batch_ids, dtype=mx.int32)  # [B, L]
            logits = self(ids_mx)                           # [B, L, 64]
            mx.eval(logits)

            log_probs = np.array(nn.log_softmax(logits.astype(mx.float32), axis=-1))  # [B, L, 64]
            for b_idx, pos in enumerate(batch_pos):
                pos_to_lp[pos] = log_probs[b_idx, pos + 1]  # log-probs at masked pos

        scores = []
        for pos, wt_aa, mut_aa in mutations:
            if pos not in pos_to_lp or wt_aa not in AA_TO_ID or mut_aa not in AA_TO_ID:
                scores.append(float("nan"))
                continue
            lp = pos_to_lp[pos]
            scores.append(float(lp[AA_TO_ID[mut_aa]] - lp[AA_TO_ID[wt_aa]]))
        return scores

    # ------------------------------------------------------------------
    # Optimization helpers (levels 5-10)
    # ------------------------------------------------------------------

    def _apply_qkv_fusion(self) -> None:
        """Replace separate q/k/v projections and gate/up projections with
        fused counterparts.  Weights are concatenated along axis=0 (output dim).
        The layer norms (ln, q_norm, k_norm) and rope are copied by reference.
        """
        for block in self.layers:
            old_attn = block.attn
            old_ffn  = block.ffn

            # --- fused attention ---
            # q_proj.weight shape: [hidden, hidden] (out=in for square projection)
            attn_hidden = old_attn.q_proj.weight.shape[0]  # out_features = hidden
            fused_attn = EsmcAttentionFused(
                hidden  = attn_hidden,
                n_heads = old_attn.n_heads,
            )
            # Concat Q, K, V weights: each is [hidden, hidden], result [3*hidden, hidden]
            fused_attn.qkv_proj.weight = mx.concatenate(
                [old_attn.q_proj.weight, old_attn.k_proj.weight, old_attn.v_proj.weight],
                axis=0,
            )
            fused_attn.ln      = old_attn.ln
            fused_attn.q_norm  = old_attn.q_norm
            fused_attn.k_norm  = old_attn.k_norm
            fused_attn.o_proj  = old_attn.o_proj
            fused_attn.rope    = old_attn.rope
            block.attn = fused_attn

            # --- fused FFN ---
            # gate_proj.weight shape: [intermediate, hidden]
            intermediate = old_ffn.gate_proj.weight.shape[0]
            hidden_dim   = old_ffn.gate_proj.weight.shape[1]
            fused_ffn = SwiGLUFFNFused(
                hidden       = hidden_dim,
                intermediate = intermediate,
            )
            # Concat gate, up weights: each is [intermediate, hidden], result [2*intermediate, hidden]
            fused_ffn.gate_up_proj.weight = mx.concatenate(
                [old_ffn.gate_proj.weight, old_ffn.up_proj.weight],
                axis=0,
            )
            fused_ffn.ln        = old_ffn.ln
            fused_ffn.down_proj = old_ffn.down_proj
            block.ffn = fused_ffn

            # The block no longer uses the old fused-residual path (opt=4)
            block._use_fused = False

        mx.eval(self.parameters())

    def _apply_fast_layer_norm(self) -> None:
        """Replace every nn.LayerNorm in the transformer blocks and final norm
        with FastLayerNorm backed by mx.fast.layer_norm.

        LM-head LayerNorm is also replaced.  The weight/bias tensors are
        transferred so no re-loading is needed.
        """

        def _convert_ln(ln_module: nn.LayerNorm) -> FastLayerNorm:
            dims = ln_module.weight.shape[0]
            has_bias = getattr(ln_module, "bias", None) is not None
            fast_ln = FastLayerNorm(dims, eps=getattr(ln_module, "eps", 1e-5), bias=has_bias)
            fast_ln.weight = ln_module.weight
            if has_bias:
                fast_ln.bias = ln_module.bias
            return fast_ln

        for block in self.layers:
            # Attention and FFN pre-norms
            block.attn.ln = _convert_ln(block.attn.ln)
            block.ffn.ln  = _convert_ln(block.ffn.ln)
            # QK-Norms (no bias)
            block.attn.q_norm = _convert_ln(block.attn.q_norm)
            block.attn.k_norm = _convert_ln(block.attn.k_norm)

        # Final encoder norm (no bias)
        self.norm = _convert_ln(self.norm)

        # LM-head norm
        self.lm_head.layer_norm = _convert_ln(self.lm_head.layer_norm)

        mx.eval(self.parameters())

    def _apply_quant_ffn(self, bits: int, group_size: int,
                         mode: str = "affine",
                         layers_filter: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj",
                                                            "gate_up_proj")) -> None:
        """Quantize only FFN linear layers; attention stays in BF16."""
        nn.quantize(
            self,
            bits=bits,
            group_size=group_size,
            mode=mode,
            class_predicate=lambda path, module: (
                isinstance(module, nn.Linear) and
                any(name in str(path) for name in layers_filter)
            ),
        )
        mx.eval(self.parameters())

    # ------------------------------------------------------------------
    # Optimization API
    # ------------------------------------------------------------------

    def optimize(self, level: int) -> "EsmcMLX":
        """Apply optimization level in-place. Returns self."""
        if level >= 2:
            self.set_dtype(mx.bfloat16)
            mx.eval(self.parameters())

        if level >= 4 and level not in range(5, 11):
            # opt=4 only: fused residual+LN Metal kernel path
            for block in self.layers:
                block._use_fused = True
                block.attn._use_fused = True
                block.ffn._use_fused = True

        # ---- levels 5-10: QKV + SwiGLU fusion (and further per-level opts) ----

        if level == 5:
            # bf16 already applied above; apply fusion then compile
            self._apply_qkv_fusion()

        elif level == 6:
            # QKV fusion + bits=6 affine FFN quantization
            self._apply_qkv_fusion()
            self._apply_quant_ffn(bits=6, group_size=64, mode="affine",
                                  layers_filter=("gate_up_proj", "down_proj"))

        elif level == 7:
            # QKV fusion + mxfp4 FFN quantization (group_size=32 required for mxfp4)
            self._apply_qkv_fusion()
            self._apply_quant_ffn(bits=4, group_size=32, mode="mxfp4",
                                  layers_filter=("gate_up_proj", "down_proj"))

        elif level == 8:
            # QKV fusion + bits=3 down_proj only (largest FFN weight: intermediate→hidden)
            self._apply_qkv_fusion()
            self._apply_quant_ffn(bits=3, group_size=64, mode="affine",
                                  layers_filter=("down_proj",))

        elif level == 9:
            # bf16+compile + fast layer_norm (no QKV fusion)
            self._apply_fast_layer_norm()

        elif level == 10:
            # Kitchen sink: QKV fusion + SwiGLU fusion + bits=6 FFN + FastLayerNorm
            self._apply_qkv_fusion()
            self._apply_quant_ffn(bits=6, group_size=64, mode="affine",
                                  layers_filter=("gate_up_proj", "down_proj"))
            self._apply_fast_layer_norm()

        # Compile for levels that benefit from JIT (all except pure-bf16 level 2)
        if level in (1, 3, 4, 5, 6, 7, 8, 9, 10):
            dummy = mx.zeros((1, 4), dtype=mx.int32)
            out = self._raw_forward(dummy); mx.eval(out)
            self._compiled_fn = mx.compile(self._raw_forward)
            out = self._compiled_fn(dummy); mx.eval(out)

        self._opt_level = level
        return self

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "biohub/ESMC-300M",
        opt_level: int = 0,
    ) -> "EsmcMLX":
        if repo_id not in cls.CONFIGS:
            raise ValueError(f"Unknown: {repo_id!r}. Known: {list(cls.CONFIGS)}")
        cfg = cls.CONFIGS[repo_id]
        # use_fused_ln (Metal kernel path) only applies to level 4
        use_fused = (opt_level == 4)
        model = cls(**cfg, use_fused_ln=use_fused)
        local_dir = Path(snapshot_download(repo_id, ignore_patterns=["*.msgpack"]))
        weights = _load_safetensors(local_dir)
        _load_weights(model, weights, cfg["n_layers"])
        mx.eval(model.parameters())
        if opt_level > 0:
            model.optimize(opt_level)
        return model


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _load_safetensors(local_dir: Path) -> dict[str, mx.array]:
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("pip install safetensors")
    weights: dict[str, mx.array] = {}
    for shard in sorted(local_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="numpy") as f:
            for key in f.keys():
                weights[key] = mx.array(f.get_tensor(key))
    return weights


def _load_weights(model: EsmcMLX, w: dict[str, mx.array], n_layers: int):
    def get(key: str) -> mx.array:
        if key not in w:
            raise KeyError(f"Weight not found: {key!r}")
        return w[key]

    model.embed_tokens.weight = get("esmc.embed_tokens.weight")

    for i, block in enumerate(model.layers):
        p = f"esmc.layers.{i}"
        a = block.attn
        a.ln.weight     = get(f"{p}.input_layernorm.weight")
        a.ln.bias       = get(f"{p}.input_layernorm.bias")
        a.q_proj.weight = get(f"{p}.self_attn.q_proj.weight")
        a.k_proj.weight = get(f"{p}.self_attn.k_proj.weight")
        a.v_proj.weight = get(f"{p}.self_attn.v_proj.weight")
        a.o_proj.weight = get(f"{p}.self_attn.o_proj.weight")
        a.q_norm.weight = get(f"{p}.self_attn.q_norm.weight")
        a.k_norm.weight = get(f"{p}.self_attn.k_norm.weight")

        f_ = block.ffn
        f_.ln.weight        = get(f"{p}.post_attention_layernorm.weight")
        f_.ln.bias          = get(f"{p}.post_attention_layernorm.bias")
        f_.gate_proj.weight = get(f"{p}.mlp.gate_proj.weight")
        f_.up_proj.weight   = get(f"{p}.mlp.up_proj.weight")
        f_.down_proj.weight = get(f"{p}.mlp.down_proj.weight")

    model.norm.weight = get("esmc.norm.weight")

    model.lm_head.dense.weight      = get("lm_head.dense.weight")
    model.lm_head.dense.bias        = get("lm_head.dense.bias")
    model.lm_head.layer_norm.weight = get("lm_head.layer_norm.weight")
    model.lm_head.layer_norm.bias   = get("lm_head.layer_norm.bias")
    model.lm_head.decoder.weight    = get("lm_head.decoder.weight")
    model.lm_head.decoder.bias      = get("lm_head.decoder.bias")


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def verify_against_pytorch(seq: str = "MKTAYIAKQRQISFVKSHFSRQ", tol: float = 1e-2):
    import torch
    from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
    from esm.models.esmc.config import ESMC_300M_HF_REPO

    pt = EsmcForMaskedLM.from_pretrained(
        ESMC_300M_HF_REPO, device=torch.device("cpu"),
        dtype=torch.float32, attn_implementation="sdpa",
    )
    pt.eval()
    tok = EsmcTokenizer.from_pretrained(ESMC_300M_HF_REPO)
    enc = tok(seq, return_tensors="pt")
    with torch.no_grad():
        pt_logits = pt(**enc).logits.numpy()

    mlx_model = EsmcMLX.from_pretrained("biohub/ESMC-300M")
    ids = mx.array(enc["input_ids"].numpy())
    mlx_logits = np.array(mlx_model(ids))

    mae = float(np.abs(pt_logits - mlx_logits).max())
    assert mae < tol, f"Logits diverge: {mae:.4f} > {tol}"
    return mae
