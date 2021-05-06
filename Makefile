.PHONY: docs

docs:
	rm -rf docs/_build/
	find docs/api ! -name 'index.rst' -type f -exec rm -f {} +
	pip install -qr docs/requirements.txt
	jb build docs

typestubs:
	python -m napari.view_layers
	python -m napari.components.viewer_model
