# napari — AI Coding Agent Instructions

napari is a fast, interactive, multi-dimensional image viewer for Python.
Built on Qt (GUI), vispy (GPU rendering), and the scientific Python stack.

This is a worktree for the histogram PR (#8391) implementation and UX overhaul.

## Quick reference

### Environment (uv only — no pip, no conda)

```sh
uv venv -p 3.13 --clear         # recreate .venv
uv pip install -e ".[optional]" --group dev
prek install                     # install pre-commit hooks via prek
```

### Testing

```sh
uv run pytest                              # all tests (requires Qt)
uv run pytest -m "not qt"                  # headless tests only
uv run pytest napari/_tests/               # core tests
uv run tox -e mypy                         # type checking
```

### Code quality

```sh
uv run prek -a -q                # run all pre-commit hooks (ruff lint+format, etc.)
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
```

### Build docs

```sh
uv run make html                 # or `html-noplot` to skip gallery
uv run make html-live            # auto-rebuild on save
```

### Build settings schema

```sh
uv run make settings-schema
```

## Key conventions

- **Qt bindings**: Always use `from qtpy` for Qt imports; never import PyQt/PySide directly.
- **Import-linter**: Enforced via pre-commit (`import-linter` hook, manual stage).
- **Lazy loading**: `napari/__init__.py` uses `lazy_loader`.
- **Versioning**: EffVer via `setuptools_scm`.
- **Environment management**: Use `uv` exclusively — no `pip`, `conda`, or other package managers.
- **Pre-commit**: Use `prek` (not `pre-commit`) — run with `uv run prek -a -q`.
