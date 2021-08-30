.PHONY: docs typestubs pre watch dist settings-schema

docs:
	rm -rf docs/_build/
	find docs/api ! -name 'index.rst' -type f -exec rm -f {} +
	pip install -qr docs/requirements.txt
	python docs/update_docs.py
	NAPARI_APPLICATION_IPY_INTERACTIVE=0
	jb build docs
	unset NAPARI_APPLICATION_IPY_INTERACTIVE

typestubs:
	python -m napari.utils.stubgen

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
