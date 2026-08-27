#!/usr/bin/env bash
set -euo pipefail

mkdir -p docs/_static

cffconvert \
  --infile CITATION.cff \
  --outfile docs/_static/citation.bib \
  --format bibtex

cat > docs/_static/citation.tex <<'EOF'
\documentclass{article}
\usepackage[backend=biber,style=authoryear]{biblatex}
\addbibresource{citation.bib}
\pagestyle{empty}

\begin{document}
\nocite{*}
\printbibliography[heading=none]
\end{document}
EOF

cd docs/_static

pdflatex -interaction=nonstopmode citation.tex
biber citation
pdflatex -interaction=nonstopmode citation.tex
pdflatex -interaction=nonstopmode citation.tex

pdftotext -layout citation.pdf citation.txt
