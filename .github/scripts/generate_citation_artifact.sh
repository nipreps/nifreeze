#!/usr/bin/env bash
set -euo pipefail

echo "Extracting citation.bib via cffr..."

Rscript -e '
  library(cffr)
  bib <- cff_extract_to_bibtex("CITATION.cff", what = "all")
  writeLines(toBibtex(bib), "docs/_static/citation.bib")
'

# Sanitize special characters in the generated .bib file for LaTeX compatibility
# Escapes unescaped & to \& and # to \#
sed -i 's/#/\\#/g' docs/_static/citation.bib
sed -i 's/&/\\&/g' docs/_static/citation.bib

echo "Generating LaTeX citation preview..."

cat > docs/_static/citation.tex <<'EOF'
\documentclass{article}
\usepackage[backend=biber,style=authoryear,maxbibnames=99]{biblatex}
\addbibresource{citation.bib}
\pagestyle{empty}

\begin{document}
\nocite{*}
\printbibliography[heading=none]
\end{document}
EOF

cd docs/_static

# Clean old aux/bbl files to prevent stale state issues
rm -f citation.aux citation.bbl citation.blg citation.log citation.out citation.pdf citation.txt

pdflatex -interaction=nonstopmode citation.tex
biber citation
pdflatex -interaction=nonstopmode citation.tex
pdflatex -interaction=nonstopmode citation.tex

pdftotext -layout citation.pdf citation.txt

# Strip the trailing form-feed character (\x0c) introduced by pdftotext layout mode
tr -d '\014' < citation.txt > citation_clean.txt
mv citation_clean.txt citation.txt

echo "Citation build complete!"
