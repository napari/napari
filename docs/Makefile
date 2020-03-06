# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

.PHONY: buildapi
buildapi:
	rm -rf source/api/*
	sphinx-apidoc -f -o source/api ../napari ../napari/*tests* ../napari/**/*tests* ../napari/**/**/*tests* ../napari/utils/colormaps/vendored

.PHONY: clean
clean:
	rm -rf source/api/*
	rm -rf $(BUILDDIR)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	git fetch --tags  # update tags so version info is correct
	make buildapi
	rm -rf source/release
	cp -R release source/
	rm -f source/release/generate_release_notes.py
	rm -f source/release/release_dev.rst
	rm -f source/release/release_template.rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
