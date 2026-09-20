"""
ESMC-300M / ESMC-600M inference in MLX (Apple Silicon, native Metal).

Loads weights from the cached HuggingFace checkpoint (biohub/ESMC-300M or
biohub/ESMC-600M) and runs the full forward pass in MLX for native Metal GPU
acceleration on M-series chips.

Architecture: pre-LN transformer · RoPE (base=10000) · QK-Norm · SwiGLU FFN
  300M: 30 layers, hidden=960,  heads=15, intermediate=2560
  600M: 36 layers, hidden=1152, heads=16, intermediate=3072

Usage:
    from scoring.esmc.mlx_esmc import EsmcMLX
    model = EsmcMLX.from_pretrained("biohub/ESMC-300M")
    # input_ids: mx.array of shape [B, L], token ids from EsmcTokenizer
    logits = model(input_ids)   # [B, L, 64]
    log_probs = mx.log_softmax(logits, axis=-1)
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """Pre-LN SwiGLU feed-forward block.

    Safetensors keys (per layer i):
        layers.{i}.post_attention_layernorm.{weight,bias}
        layers.{i}.mlp.gate_proj.weight   [intermediate, hidden]
        layers.{i}.mlp.up_proj.weight     [intermediate, hidden]
        layers.{i}.mlp.down_proj.weight   [hidden, intermediate]
    """

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.ln(x)
        return self.down_proj(nn.silu(self.gate_proj(h)) * self.up_proj(h))


class EsmcAttention(nn.Module):
    """Pre-LN multi-head self-attention with QK-Norm and RoPE.

    Safetensors keys (per layer i):
        layers.{i}.input_layernorm.{weight,bias}
        layers.{i}.self_attn.{q,k,v}_proj.weight   [hidden, hidden]  (no bias)
        layers.{i}.self_attn.o_proj.weight          [hidden, hidden]  (no bias)
        layers.{i}.self_attn.q_norm.weight          [hidden]          (no bias)
        layers.{i}.self_attn.k_norm.weight          [hidden]          (no bias)
    """

    def __init__(self, hidden: int, n_heads: int):
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

    def __call__(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape
        h = self.ln(x)

        q = self.q_norm(self.q_proj(h))   # [B, L, hidden]
        k = self.k_norm(self.k_proj(h))
        v = self.v_proj(h)

        # [B, L, H*D] -> [B, H, L, D]
        def to_heads(t: mx.array) -> mx.array:
            return t.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        q = self.rope(q)
        k = self.rope(k)

        # [B, H, L, D]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # [B, H, L, D] -> [B, L, hidden]
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class EsmcBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, intermediate: int,
                 scaling_factor: float = 1.0):
        super().__init__()
        self.attn = EsmcAttention(hidden, n_heads)
        self.ffn  = SwiGLUFFN(hidden, intermediate)
        self.scaling_factor = scaling_factor

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(x) / self.scaling_factor
        x = x + self.ffn(x) / self.scaling_factor
        return x


class EsmcLMHead(nn.Module):
    """Sequential: Linear → GELU → LayerNorm → Linear.

    Safetensors keys:
        lm_head.dense.{weight,bias}
        lm_head.layer_norm.{weight,bias}
        lm_head.decoder.{weight,bias}
    """

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
    """ESMC in MLX — supports both 300M and 600M checkpoints."""

    CONFIGS = {
        "biohub/ESMC-300M": dict(hidden=960,  n_heads=15, n_layers=30, intermediate=2560),
        "biohub/ESMC-600M": dict(hidden=1152, n_heads=16, n_layers=36, intermediate=3072),
    }

    def __init__(self, hidden: int, n_heads: int, n_layers: int,
                 intermediate: int, vocab: int = 64):
        super().__init__()
        # residue scaling: sqrt(n_layers/36) — ESM3 scheme, matches scale_residue=True
        scaling_factor = math.sqrt(n_layers / 36)
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = [EsmcBlock(hidden, n_heads, intermediate, scaling_factor)
                       for _ in range(n_layers)]
        self.norm   = nn.LayerNorm(hidden, bias=False)
        self.lm_head = EsmcLMHead(hidden, vocab)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.embed_tokens(input_ids)
        for block in self.layers:
            x = block(x)
        return self.lm_head(self.norm(x))

    @classmethod
    def from_pretrained(cls, repo_id: str = "biohub/ESMC-300M") -> "EsmcMLX":
        if repo_id not in cls.CONFIGS:
            raise ValueError(f"Unknown: {repo_id!r}. Known: {list(cls.CONFIGS)}")
        cfg = cls.CONFIGS[repo_id]
        model = cls(**cfg)
        local_dir = Path(snapshot_download(repo_id, ignore_patterns=["*.msgpack"]))
        weights = _load_safetensors(local_dir)
        _load_weights(model, weights, cfg["n_layers"])
        mx.eval(model.parameters())
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
            raise KeyError(f"Weight not found in checkpoint: {key!r}")
        return w[key]

    model.embed_tokens.weight = get("esmc.embed_tokens.weight")

    for i, block in enumerate(model.layers):
        p = f"esmc.layers.{i}"

        a = block.attn
        a.ln.weight    = get(f"{p}.input_layernorm.weight")
        a.ln.bias      = get(f"{p}.input_layernorm.bias")
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
    """Runs both forward passes, asserts max absolute error < tol."""
    import torch
    from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
    from esm.models.esmc.config import ESMC_300M_HF_REPO

    print("PyTorch reference (CPU float32)...")
    pt = EsmcForMaskedLM.from_pretrained(
        ESMC_300M_HF_REPO, device=torch.device("cpu"),
        dtype=torch.float32, attn_implementation="sdpa",
    )
    pt.eval()
    tok = EsmcTokenizer.from_pretrained(ESMC_300M_HF_REPO)
    enc = tok(seq, return_tensors="pt")
    with torch.no_grad():
        pt_logits = pt(**enc).logits.numpy()          # (1, L+2, 64)

    print("MLX model (Metal)...")
    mlx_model = EsmcMLX.from_pretrained("biohub/ESMC-300M")
    ids = mx.array(enc["input_ids"].numpy())
    mlx_logits = np.array(mlx_model(ids))             # (1, L+2, 64)

    mae = float(np.abs(pt_logits - mlx_logits).max())
    print(f"Max absolute error: {mae:.6f}  (tol={tol})")
    assert mae < tol, f"Logits diverge: {mae:.4f} > {tol}"
    print("PASS")
    return mae


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

def benchmark(seq: str, n_runs: int = 20):
    import time
    import torch
    from esm.models.esmc import ESMC, EsmcTokenizer
    from esm.models.esmc.config import ESMC_300M_HF_REPO
    from esm.sdk.api import ESMProtein, LogitsConfig

    print(f"\nBenchmark  L={len(seq)}  n_runs={n_runs}")
    print("-" * 50)

    tok = EsmcTokenizer.from_pretrained(ESMC_300M_HF_REPO)
    enc = tok(seq, return_tensors="pt")

    # -- MLX --
    mlx_model = EsmcMLX.from_pretrained("biohub/ESMC-300M")
    ids = mx.array(enc["input_ids"].numpy())
    out = mlx_model(ids); mx.eval(out)                 # warm-up + Metal compile

    t0 = time.perf_counter()
    for _ in range(n_runs):
        out = mlx_model(ids)
        mx.eval(out)
    mlx_ms = (time.perf_counter() - t0) / n_runs * 1000

    # -- PyTorch MPS --
    pt_model = ESMC.from_pretrained("esmc_300m", device=torch.device("mps"),
                                    use_flash_attn=False)
    pt_model.eval()
    protein = ESMProtein(sequence=seq)
    encoded = pt_model.encode(protein)
    with torch.no_grad():
        pt_model.logits(encoded, LogitsConfig(sequence=True))
    torch.mps.synchronize()                            # warm-up

    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            pt_model.logits(encoded, LogitsConfig(sequence=True))
        torch.mps.synchronize()
    mps_ms = (time.perf_counter() - t0) / n_runs * 1000

    print(f"MLX  (Metal):  {mlx_ms:.1f} ms/pass")
    print(f"PyTorch (MPS): {mps_ms:.1f} ms/pass")
    print(f"Speedup:       {mps_ms / mlx_ms:.2f}x")
    return {"mlx_ms": mlx_ms, "mps_ms": mps_ms, "speedup": mps_ms / mlx_ms}


if __name__ == "__main__":
    print("=== MLX ESMC-300M ===\n")
    mae = verify_against_pytorch()
    print()
    seq = "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"
    benchmark(seq, n_runs=20)
