.PHONY: docs

docs:
	rm -rf docs/_build/html
	find docs/api ! -name 'index.rst' -type f -exec rm -f {} +
	pip install -qr docs/requirements.txt
	jb build docs
