.PHONY: typestubs pre watch dist settings-schema

typestubs:
	python -m napari.utils.stubgen

# note: much faster to run mypy as daemon,
# dmypy run -- ...
# https://mypy.readthedocs.io/en/stable/mypy_daemon.html
typecheck:
	mypy napari/settings napari/types.py napari/plugins

check-manifest:
	pip install -U check-manifest
	check-manifest

dist: typestubs check-manifest
	pip install -U build
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
