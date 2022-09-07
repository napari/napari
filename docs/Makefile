.PHONY: typestubs pre watch dist settings-schema

clean:
	rm -rf _build/
	rm -rf api/napari*.rst
	rm -rf gallery/*
	rm -rf _tags

docs:
	python _scripts/prep_docs.py
	NAPARI_APPLICATION_IPY_INTERACTIVE=0 sphinx-build -b html . ./_build

html:
	python _scripts/prep_docs.py
	NAPARI_APPLICATION_IPY_INTERACTIVE=0 sphinx-build -b html . ./_build

html-noplot:
	python _scripts/prep_docs.py
	NAPARI_APPLICATION_IPY_INTERACTIVE=0 sphinx-build -D plot_gallery=0 -b html . ./_build
