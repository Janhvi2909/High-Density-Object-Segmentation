#!/bin/bash
# Compile LaTeX report

# Navigate to report directory
cd "$(dirname "$0")"

# Compile with pdflatex
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# Clean auxiliary files
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

echo "Report compiled: main.pdf"
