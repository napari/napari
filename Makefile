docs:
	pip install -q -r requirements/docs.txt
	make -C docs clean
	make -C docs html
.PHONY: docs
