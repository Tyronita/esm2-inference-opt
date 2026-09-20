"""
ESMC-300M local inference on Apple Silicon (MPS).

Uses biohub/ESMC-300M via the EvolutionaryScale SDK.
Run with: .venv-esmc/bin/python3 scoring/esmc/local_300m.py

Attention: the `flash_attn` library is CUDA-only and is not installed here.
When attn_implementation="flash_attention_2" is requested without flash_attn,
HuggingFace silently falls back to SDPA. use_flash_attn=False is explicit but
makes no practical difference on MPS — both paths land on SDPA.

dtype: bfloat16 on MPS (matches GPU behaviour, saves ~600MB vs float32).

Published ProteinGym ground truth (ρ target): 0.406 (rank #49/103 models).
"""
import time
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_esmc_300m(device: torch.device) -> ESMC:
    print(f"Loading biohub/ESMC-300M on {device} ...")
    t0 = time.perf_counter()
    model = ESMC.from_pretrained("esmc_300m", device=device, use_flash_attn=False)
    model.eval()
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    param_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"  Params: {param_m:.0f}M")
    return model


def score_sequence(model: ESMC, seq: str, device: torch.device) -> dict:
    """
    Masked-marginals score for every position in seq.
    Returns per-position log-likelihoods (L, vocab_size).
    """
    protein = ESMProtein(sequence=seq)
    t0 = time.perf_counter()

    with torch.no_grad():
        encoded = model.encode(protein)
        logits_out = model.logits(
            encoded,
            LogitsConfig(sequence=True, return_embeddings=False),
        )

    elapsed = time.perf_counter() - t0
    logits = logits_out.logits.sequence  # (1, L+2, vocab)
    return {"logits": logits, "elapsed_s": elapsed}


def demo():
    device = get_device()
    model = load_esmc_300m(device)

    test_seqs = [
        ("SNCA (short)", "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"),
        ("GFP chromophore region", "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"),
    ]

    for name, seq in test_seqs:
        print(f"\n{name}  (L={len(seq)})")
        result = score_sequence(model, seq, device)
        logits = result["logits"]
        print(f"  Logits shape: {logits.shape}  dtype: {logits.dtype}")
        print(f"  Elapsed: {result['elapsed_s']*1000:.0f} ms")
        # Top token at position 1 (first real residue after BOS)
        top_id = logits[0, 1].argmax().item()
        print(f"  Top token at pos 1: id={top_id}")

    print("\nESMC-300M local inference OK")


if __name__ == "__main__":
    demo()
