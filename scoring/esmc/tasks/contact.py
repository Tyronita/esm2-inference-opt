"""
Contact prediction from ESMC attention maps.

Method: Rao et al. 2020 (ESM-1b) — average all-layer, all-head attention,
symmetrise, apply APC correction, report precision at L (P@L) for long-range
contacts (|i-j| > 6, Cβ distance < 8 Å).

Datasets fetched on demand from RCSB PDB.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Ground-truth contacts from PDB
# ---------------------------------------------------------------------------

def _fetch_pdb(pdb_id: str) -> str:
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode()


def _cb_coords(pdb_text: str, chain: str = "A") -> tuple[list[str], np.ndarray]:
    """Extract Cβ (or Cα for Gly) coordinates for a chain.

    Returns (residue_sequence, coords [L, 3]).
    """
    from Bio import PDB as bpdb  # biopython

    parser = bpdb.PDBParser(QUIET=True)
    structure = parser.get_structure("x", io.StringIO(pdb_text))
    model = structure[0]

    # pick first chain matching label, fallback to first chain
    chains = list(model.get_chains())
    target = next((c for c in chains if c.id == chain), chains[0])

    residues, coords = [], []
    aa_map = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
        "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
        "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    }
    for res in target.get_residues():
        resname = res.get_resname().strip()
        if resname not in aa_map:
            continue
        atom = "CB" if resname != "GLY" else "CA"
        if atom not in res:
            continue
        residues.append(aa_map[resname])
        coords.append(res[atom].get_vector().get_array())

    return "".join(residues), np.array(coords, dtype=np.float32)


def true_contact_map(coords: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    """Boolean (L, L) — True where Cβ distance < threshold Å."""
    diff = coords[:, None, :] - coords[None, :, :]   # (L, L, 3)
    dist = np.sqrt((diff ** 2).sum(-1))               # (L, L)
    return dist < threshold


# ---------------------------------------------------------------------------
# Predicted contacts from ESMC attention maps
# ---------------------------------------------------------------------------

def attn_contact_map(attentions: list[mx.array]) -> np.ndarray:
    """APC-corrected, symmetrised contact map from all-layer attention.

    attentions : list of (1, H, L, L) arrays, one per layer.
    Returns normalised float (L, L).
    """
    # Stack → (n_layers, H, L, L), strip BOS/EOS
    all_attn = np.stack([np.array(a[0]) for a in attentions], axis=0)  # (30, 15, L, L)
    # Average across layers and heads → (L, L)
    avg = all_attn.mean(axis=(0, 1))

    # Strip BOS / EOS (first and last position)
    avg = avg[1:-1, 1:-1]

    # Symmetrise
    sym = (avg + avg.T) / 2.0

    # APC correction
    row_mean = sym.mean(axis=1, keepdims=True)
    col_mean = sym.mean(axis=0, keepdims=True)
    global_mean = sym.mean()
    apc = sym - (row_mean * col_mean) / (global_mean + 1e-8)

    # Normalise to [0, 1]
    apc -= apc.min()
    if apc.max() > 0:
        apc /= apc.max()
    return apc


# ---------------------------------------------------------------------------
# P@L metric
# ---------------------------------------------------------------------------

def precision_at_l(pred: np.ndarray, true: np.ndarray,
                   min_sep: int = 6, k: int = 1) -> float:
    """Precision of top-k*L long-range contact predictions.

    pred / true : (L, L) symmetric matrices.
    """
    L = pred.shape[0]
    # Mask short-range and diagonal
    mask = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) >= min_sep
    upper = np.triu(mask, k=1)

    pred_vals = pred[upper]
    true_vals = true[upper]

    n_top = k * L
    top_idx = np.argsort(pred_vals)[::-1][:n_top]
    precision = true_vals[top_idx].mean()
    return float(precision)


# ---------------------------------------------------------------------------
# End-to-end task runner
# ---------------------------------------------------------------------------

@dataclass
class ContactResult:
    pdb_id: str
    chain: str
    length: int
    n_contacts: int
    p_at_l: float
    p_at_l2: float
    p_at_l5: float


def run(model, tokenizer, targets: list[tuple[str, str]] | None = None,
        verbose: bool = True) -> list[ContactResult]:
    """
    Run contact prediction on a list of (pdb_id, chain) pairs.

    Default targets: 5 CASP15 proteins of varying length.
    """
    if targets is None:
        # Classic mixed-topology proteins with well-resolved structures
        targets = [
            ("1UBQ", "A"),   # Ubiquitin — 76 aa, α/β
            ("2LZM", "A"),   # T4 lysozyme — 164 aa, helical
            ("1TEN", "A"),   # Fibronectin III — 91 aa, all-β
            ("8FBK", "A"),   # CASP15 T1109 — 218 aa, mixed
            ("8BSN", "A"),   # CASP15 T1121 — 311 aa, helical
        ]

    results = []
    for pdb_id, chain in targets:
        if verbose:
            print(f"\n{pdb_id}:{chain}", end="  ", flush=True)
        try:
            pdb_text = _fetch_pdb(pdb_id)
            seq, coords = _cb_coords(pdb_text, chain)
            L = len(seq)
            if verbose:
                print(f"L={L}", end="  ", flush=True)

            true_cm = true_contact_map(coords)
            n_contacts = int(true_cm.sum()) // 2

            # encode with attentions
            enc = tokenizer(seq, return_tensors="pt")
            ids = mx.array(enc["input_ids"].numpy())
            _, _, attentions = model.encode(ids, return_attentions=True)
            mx.eval(*[a for a in attentions if a is not None])

            pred_cm = attn_contact_map(attentions)

            p1  = precision_at_l(pred_cm, true_cm, k=1)
            p2  = precision_at_l(pred_cm, true_cm, k=2)
            p5  = precision_at_l(pred_cm, true_cm, k=5)

            if verbose:
                print(f"P@L={p1:.3f}  P@2L={p2:.3f}  P@5L={p5:.3f}")

            results.append(ContactResult(pdb_id, chain, L, n_contacts, p1, p2, p5))
        except Exception as e:
            if verbose:
                print(f"SKIP ({e})")

    if verbose and results:
        avg = np.mean([r.p_at_l for r in results])
        print(f"\nMean P@L across {len(results)} proteins: {avg:.3f}")

    return results
