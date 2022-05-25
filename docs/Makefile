.PHONY: typestubs pre watch dist settings-schema

clean:
	rm -rf docs/_build/
	rm -rf docs/api/napari*.rst
	rm -rf docs/gallery/*

docs:
	python _scripts/prep_docs.py
	NAPARI_APPLICATION_IPY_INTERACTIVE=0 sphinx-build -b html . ./_build

html:
	python _scripts/prep_docs.py
	NAPARI_APPLICATION_IPY_INTERACTIVE=0 sphinx-build -b html . ./_build

html-noplot:
	python _scripts/prep_docs.py
	NAPARI_APPLICATION_IPY_INTERACTIVE=0 sphinx-build -D plot_gallery=0 -b html . ./_build
