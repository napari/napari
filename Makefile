.PHONY: docs typestubs pre watch dist settings-schema

docs:
	rm -rf docs/_build/
	find docs/api ! -name 'index.rst' -type f -exec rm -f {} +
	pip install -qr docs/requirements.txt

#   some plugin docs live in npe2 for testing purposes
	rm -rf npe2
	git clone https://github.com/napari/npe2
#	remove next line after 3906	
	pip install -e ./npe2
	python npe2/_docs/render.py docs/plugins
	rm -rf npe2

	python docs/_scripts/update_preference_docs.py
	python docs/_scripts/update_event_docs.py
	NAPARI_APPLICATION_IPY_INTERACTIVE=0 jb build docs

typestubs:
	python -m napari.utils.stubgen

# note: much faster to run mypy as daemon,
# dmypy run -- ...
# https://mypy.readthedocs.io/en/stable/mypy_daemon.html
typecheck:
	mypy napari/settings napari/types.py napari/plugins napari/utils/context	

dist:
	pip install -U check-manifest build
	make typestubs
	check-manifest
	python -m build

settings-schema:
	python -m napari.settings._napari_settings

pre:
	pre-commit run -a

# If the first argument is "watch"...
ifeq (watch,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "watch"
  WATCH_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(WATCH_ARGS):;@:)
endif

# examples:
# make watch ~/Desktop/Untitled.png
# make watch -- -w animation  # -- is required for passing flags to napari

watch:
	@echo "running: napari $(WATCH_ARGS)"
	@echo "Save any file to restart napari\nCtrl-C to stop..\n" && \
		watchmedo auto-restart -R \
			--ignore-patterns="*.pyc*" -D \
			--signal SIGKILL \
			napari -- $(WATCH_ARGS) || \
		echo "please run 'pip install watchdog[watchmedo]'"
