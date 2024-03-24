"""Test app-model Qt-related providers."""

import pytest

from napari._qt._qapp_model.injection._qproviders import (
    _provide_qt_viewer_or_raise,
    _provide_window_or_raise,
)
from napari._qt.qt_main_window import Window
from napari._qt.qt_viewer import QtViewer


def test_provide_qt_viewer_or_raise(make_napari_viewer):
    """Check `_provide_qt_viewer_or_raise` raises or returns `QtViewer`."""
    # raises when no QtViewer
    with pytest.raises(
        RuntimeError, match='No current `QtViewer` found. test'
    ):
        _provide_qt_viewer_or_raise(msg='test')

    # create QtViewer
    make_napari_viewer()
    viewer = _provide_qt_viewer_or_raise()
    assert isinstance(viewer, QtViewer)


def test_provide_window_or_raise(make_napari_viewer):
    """Check `_provide_window_or_raise` raises or returns `Window`."""
    # raises when no Window
    with pytest.raises(RuntimeError, match='No current `Window` found. test'):
        _provide_window_or_raise(msg='test')

    # create viewer (and Window)
    make_napari_viewer()
    viewer = _provide_window_or_raise()
    assert isinstance(viewer, Window)
