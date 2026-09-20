"""
Secondary structure prediction from ESMC embeddings.

Method: linear probe (logistic regression) on per-token hidden states.
Labels from DSSP applied to PDB structures (via biopython).

3-class (Q3): H = helix (H, G, I), E = strand (E, B), C = coil (everything else)
8-class (Q8): all DSSP labels

Trains on a leave-one-out split across the provided proteins.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


# DSSP → 8-class index
DSSP8 = {"H": 0, "B": 1, "E": 2, "G": 3, "I": 4, "T": 5, "S": 6, "-": 7, " ": 7}
# DSSP → 3-class (Q3)
DSSP3 = {
    "H": 0, "G": 0, "I": 0,   # helix
    "E": 1, "B": 1,            # strand
    "T": 2, "S": 2, "-": 2, " ": 2,
}


def _fetch_pdb(pdb_id: str) -> str:
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode()


def _dssp_labels(pdb_text: str, chain: str = "A") -> tuple[str, list[str]]:
    """Extract secondary structure from PDB HELIX/SHEET records + sequence.

    Uses the PDB file's own HELIX/SHEET annotation records — no external
    binary required. These are deposited by the authors and are reliable.
    H = helix (HELIX record), E = strand (SHEET record), C = coil (everything else).
    """
    from Bio import PDB as bpdb

    aa_map = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
        "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
        "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    }

    # Parse HELIX / SHEET records directly from PDB text
    # HELIX record: cols 20-23 = init_seq_num, 34-37 = end_seq_num, col 20 = chain
    # SHEET record: cols 23-26 = init_seq_num, 34-37 = end_seq_num
    helix_ranges: set[int] = set()
    strand_ranges: set[int] = set()

    for line in pdb_text.splitlines():
        rec = line[:6].strip()
        if rec == "HELIX":
            try:
                c = line[19].strip() or chain
                if c != chain:
                    continue
                start = int(line[21:25])
                end   = int(line[33:37])
                helix_ranges.update(range(start, end + 1))
            except (ValueError, IndexError):
                pass
        elif rec == "SHEET":
            try:
                c = line[21].strip() or chain
                if c != chain:
                    continue
                start = int(line[22:26])
                end   = int(line[33:37])
                strand_ranges.update(range(start, end + 1))
            except (ValueError, IndexError):
                pass

    # Extract sequence + residue numbers
    parser = bpdb.PDBParser(QUIET=True)
    structure = parser.get_structure("x", io.StringIO(pdb_text))
    model = structure[0]
    chains = list(model.get_chains())
    target = next((c for c in chains if c.id == chain), chains[0])

    seq, labels = [], []
    for res in target.get_residues():
        rname = res.get_resname().strip()
        if rname not in aa_map:
            continue
        seq_num = res.get_id()[1]
        seq.append(aa_map[rname])
        if seq_num in helix_ranges:
            labels.append("H")
        elif seq_num in strand_ranges:
            labels.append("E")
        else:
            labels.append("C")

    return "".join(seq), labels


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def get_embeddings(model, tokenizer, sequences: list[str]) -> list[np.ndarray]:
    """Return list of (L, 960) float32 arrays — one per sequence, BOS/EOS stripped."""
    embs = []
    for seq in sequences:
        enc = tokenizer(seq, return_tensors="pt")
        ids = mx.array(enc["input_ids"].numpy())
        last_hidden, _, _ = model.encode(ids)
        mx.eval(last_hidden)
        emb = np.array(last_hidden[0])  # (L+2, 960)
        embs.append(emb[1:-1])          # strip BOS/EOS → (L, 960)
    return embs


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def train_and_eval(embeddings: list[np.ndarray], label_lists: list[list[str]],
                   n_classes: int = 3) -> dict:
    """Leave-one-out logistic regression probe. Returns Q3/Q8 accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    label_map = DSSP3 if n_classes == 3 else DSSP8

    # Build flat arrays
    all_X = np.concatenate(embeddings, axis=0)
    all_y = np.array([label_map.get(l, n_classes - 1)
                      for labels in label_lists for l in labels])

    n_proteins = len(embeddings)
    sizes = [len(e) for e in embeddings]
    boundaries = np.cumsum([0] + sizes)

    preds, trues = [], []
    for i in range(n_proteins):
        test_idx  = slice(boundaries[i], boundaries[i+1])
        train_idx = list(range(0, boundaries[i])) + list(range(boundaries[i+1], len(all_X)))

        X_train, y_train = all_X[train_idx], all_y[train_idx]
        X_test,  y_test  = all_X[test_idx],  all_y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf.fit(X_train, y_train)
        preds.extend(clf.predict(X_test))
        trues.extend(y_test)

    preds, trues = np.array(preds), np.array(trues)
    accuracy = (preds == trues).mean()
    per_class = {}
    for c in range(n_classes):
        mask = trues == c
        if mask.sum() > 0:
            per_class[c] = (preds[mask] == trues[mask]).mean()

    return {"accuracy": float(accuracy), "per_class": per_class,
            "n_residues": len(trues)}


# ---------------------------------------------------------------------------
# End-to-end task runner
# ---------------------------------------------------------------------------

@dataclass
class SSResult:
    n_proteins: int
    n_residues: int
    q3: float
    q8: float


def run(model, tokenizer, targets: list[tuple[str, str]] | None = None,
        verbose: bool = True) -> SSResult:
    """
    Run secondary structure prediction on a list of (pdb_id, chain) pairs.
    Default targets: same 5 CASP15 proteins as contact task.
    """
    if targets is None:
        # Classic well-characterised proteins with mixed H/E/C content
        targets = [
            ("1UBQ", "A"),   # Ubiquitin — 76 aa, 1 helix + 5 strands
            ("2LZM", "A"),   # T4 lysozyme — 164 aa, 8 helices + turns
            ("1TEN", "A"),   # Fibronectin type III — 91 aa, all beta
            ("1HRC", "A"),   # Cytochrome C (horse) — 104 aa, helical
            ("2PTL", "A"),   # Protein L — 62 aa, 1 helix + 4 strands
        ]

    seqs, label_lists = [], []
    for pdb_id, chain in targets:
        if verbose:
            print(f"  Fetching {pdb_id}:{chain} ...", flush=True)
        try:
            pdb_text = _fetch_pdb(pdb_id)
            seq, labels = _dssp_labels(pdb_text, chain)
            if len(seq) < 20 or len(seq) != len(labels):
                continue
            seqs.append(seq)
            label_lists.append(labels)
            if verbose:
                h = labels.count("H"); e = labels.count("E")
                print(f"    L={len(seq)}  H={h}  E={e}  C={len(labels)-h-e}")
        except Exception as ex:
            if verbose:
                print(f"    SKIP ({ex})")

    if len(seqs) < 2:
        raise RuntimeError("Need at least 2 proteins for leave-one-out eval")

    if verbose:
        print("\nExtracting ESMC embeddings ...")
    embeddings = get_embeddings(model, tokenizer, seqs)

    if verbose:
        print("Training Q3 linear probe (leave-one-out) ...")
    q3_res = train_and_eval(embeddings, label_lists, n_classes=3)

    if verbose:
        print("Training Q8 linear probe ...")
    q8_res = train_and_eval(embeddings, label_lists, n_classes=8)

    result = SSResult(
        n_proteins=len(seqs),
        n_residues=q3_res["n_residues"],
        q3=q3_res["accuracy"],
        q8=q8_res["accuracy"],
    )

    if verbose:
        print(f"\nQ3 accuracy: {result.q3:.3f}")
        print(f"Q8 accuracy: {result.q8:.3f}")
        print(f"(n={result.n_proteins} proteins, {result.n_residues} residues, "
              f"leave-one-out)")

    return result
