.PHONY: docs

docs:
	pip install -q -r requirements/docs.txt
	make -C docs clean
	make -C docs html
	open docs/build/html/index.html
