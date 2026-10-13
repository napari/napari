"""Shared fixtures for the progressive loading tests."""

import pytest

from napari.experimental import _progressive_loading


@pytest.fixture(autouse=True)
def _no_progress_bars(monkeypatch):
    """Suppress napari's Qt progress bars for the whole directory.

    The activity-dock progress bar runs ``processEvents()`` on every
    update. That nested event processing wedges Qt timer dispatch in
    headless pytest runs on macOS: the suite stops making progress and
    the CI job is killed at its step timeout. Progress-bar cosmetics are
    not what these tests cover, and the deferred-update behavior that
    does matter is tested with an injected fake bar (see
    ``test_progress_updates_deferred``).
    """
    monkeypatch.setattr(
        _progressive_loading.ProgressiveLoader,
        '_make_progress',
        lambda self, total, description: None,
    )
