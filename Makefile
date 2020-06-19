.PHONY: docs

docs:
	make -C docs clean
	make -C docs html
