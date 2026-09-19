"""
Source integrity verifier — merkle-style SHA256 comparison.

Compares every .py file in the installed fair-esm package against
the pinned submodule at refs/fair-esm/ (SHA 2b369911).

If any file has been modified post-install, this will catch it.

Usage:
    python benchmark/verify_integrity.py
    python benchmark/verify_integrity.py --verbose
"""

import hashlib
import sys
from pathlib import Path

PINNED_SHA = "2b369911bb5b4b0dda914521b9475cad1656b2ac"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*.py"))
        if "__pycache__" not in p.parts
    }


def verify(verbose: bool = False) -> bool:
    repo_root = Path(__file__).parent.parent
    ref_root = repo_root / "refs" / "fair-esm" / "esm"

    # Locate installed ESM
    try:
        import esm as _esm
        inst_root = Path(_esm.__file__).parent
    except ImportError:
        print("FAIL  fair-esm not importable")
        return False

    ref_hashes  = collect(ref_root)
    inst_hashes = collect(inst_root)

    # Only compare files present in the reference (installed may have extras like scripts/)
    ok = True
    mismatches = []
    for rel, ref_hash in sorted(ref_hashes.items()):
        inst_hash = inst_hashes.get(rel)
        if inst_hash is None:
            mismatches.append(("MISSING", rel, ref_hash[:16], "—"))
            ok = False
        elif inst_hash != ref_hash:
            mismatches.append(("DIFFER", rel, ref_hash[:16], inst_hash[:16]))
            ok = False
        elif verbose:
            print(f"  OK  {ref_hash[:16]}  {rel}")

    # Merkle root = hash of all (path, hash) pairs in sorted order
    merkle_input = "\n".join(
        f"{rel}:{h}" for rel, h in sorted(ref_hashes.items())
    ).encode()
    merkle_root = hashlib.sha256(merkle_input).hexdigest()[:32]

    print(f"\n{'='*64}")
    print(f"  fair-esm integrity check")
    print(f"  Pinned SHA : {PINNED_SHA}")
    print(f"  Ref root   : {ref_root}")
    print(f"  Inst root  : {inst_root}")
    print(f"  Files compared : {len(ref_hashes)}")
    print(f"  Merkle root    : {merkle_root}")
    print(f"{'='*64}")

    if mismatches:
        print(f"\n  FAILURES ({len(mismatches)}):")
        for status, rel, ref_h, inst_h in mismatches:
            print(f"    {status:<8}  ref={ref_h}  inst={inst_h}  {rel}")
        print(f"\n  VERDICT: TAMPERED — installed code differs from pinned SHA\n")
    else:
        print(f"\n  VERDICT: CLEAN — {len(ref_hashes)} files match pinned SHA 2b369911\n")

    return ok


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    ok = verify(verbose=verbose)
    sys.exit(0 if ok else 1)
