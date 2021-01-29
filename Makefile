.PHONY: docs

docs:
	rm -rf docs/_build/html
	pip install -qr docs/requirements.txt
	jb build docs
