# Makefile for Sphinx documentation
#
# The version comes from the package (``python -m hatch version``, or an installed copy).

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
BUILDDIR      = _build
OUTDIR        = html
PYTHONPATH    = $(PWD)/src/

ALLSPHINXOPTS = -d $(BUILDDIR)/doctrees $(SPHINXOPTS) .

.PHONY: help clean html linkcheck doctest

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files (in BUILDDIR/OUTDIR)"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  doctest    to run all doctests embedded in the documentation"
	@echo "  clean      to remove the build and the generated API pages"

clean:
	rm -rf $(BUILDDIR)/* api/

html:
	@echo "Building docs with PYTHONPATH=$(PYTHONPATH)."
	PYTHONPATH=$(PYTHONPATH) $(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/$(OUTDIR)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/$(OUTDIR)."

linkcheck:
	PYTHONPATH=$(PYTHONPATH) $(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	PYTHONPATH=$(PYTHONPATH) $(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."
