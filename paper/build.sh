#!/usr/bin/env bash
# Build the reproducibility PDF.
# Run from the repo root: bash paper/build.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAPER="$ROOT/paper"

echo "=== 1. Generating results_table.tex from JSONL ==="
cd "$ROOT"
python paper/generate_results.py --results-dir results --paper-dir paper

echo ""
echo "=== 2. Plotting performance graphs ==="
python paper/plot_events.py --events results/events.jsonl --out paper 2>/dev/null || \
  echo "  (no events.jsonl yet — skipping plots)"

echo ""
echo "=== 3. Compiling LaTeX (3-pass) ==="
cd "$PAPER"
pdflatex -interaction=nonstopmode report.tex
bibtex report
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex

echo ""
echo "=== Done ==="
echo "PDF: $PAPER/report.pdf"
open "$PAPER/report.pdf" 2>/dev/null || true
