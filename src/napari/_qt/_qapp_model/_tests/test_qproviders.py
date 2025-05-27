"""Test app-model Qt-related providers."""

import numpy as np
import pytest
from app_model.types import Action

from napari._app_model._app import get_app_model
from napari._qt._qapp_model.injection._qproviders import (
    _provide_active_layer,
    _provide_active_layer_list,
    _provide_qt_viewer_or_raise,
    _provide_viewer,
    _provide_viewer_or_raise,
    _provide_window_or_raise,
)
from napari._qt.qt_main_window import Window
from napari._qt.qt_viewer import QtViewer
from napari.components import LayerList
from napari.layers import Image
from napari.utils._proxies import PublicOnlyProxy
from napari.viewer import Viewer


def test_publicproxy_provide_viewer(capsys, make_napari_viewer):
    """Test `_provide_viewer` outputs a `PublicOnlyProxy` when appropriate.

    Check manual (e.g., internal) `_provide_viewer` calls can disable
    `PublicOnlyProxy` via `public_proxy` parameter but `PublicOnlyProxy` is always
    used when it is used as a provider.
    """
    # No current viewer, `None` should be returned
    viewer = _provide_viewer()
    assert viewer is None

    # Create a viewer
    make_napari_viewer()
    # Ensure we can disable via `public_proxy`
    viewer = _provide_viewer(public_proxy=False)
    assert isinstance(viewer, Viewer)

    # Ensure we get a `PublicOnlyProxy` when used as a provider
    def my_viewer(viewer: Viewer) -> Viewer:
        # Allows us to check type when `Action` executed
        print(type(viewer))  # noqa: T201

    action = Action(
        id='some.command.id',
        title='some title',
        callback=my_viewer,
    )
    app = get_app_model()
    app.register_action(action)
    app.commands.execute_command('some.command.id')
    captured = capsys.readouterr()
    assert 'napari.utils._proxies.PublicOnlyProxy' in captured.out


def test_provide_viewer_or_raise(make_napari_viewer):
    """Check `_provide_viewer_or_raise` raises or returns correct `Viewer`."""
    # raises when no viewer
    with pytest.raises(RuntimeError, match='No current `Viewer` found. test'):
        _provide_viewer_or_raise(msg='test')

    # create viewer
    make_napari_viewer()
    viewer = _provide_viewer_or_raise()
    assert isinstance(viewer, Viewer)

    viewer = _provide_viewer_or_raise(public_proxy=True)
    assert isinstance(viewer, PublicOnlyProxy)


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


def test_provide_active_layer_and_layer_list(make_napari_viewer):
    """Check `_provide_active_layer/_list` returns correct object."""
    shape = (10, 10)

    viewer = make_napari_viewer()
    layer_a = Image(np.random.random(shape))
    viewer.layers.append(layer_a)

    provided_layer = _provide_active_layer()
    assert isinstance(provided_layer, Image)
    assert provided_layer.data.shape == shape

    provided_layers = _provide_active_layer_list()
    assert isinstance(provided_layers, LayerList)
    assert isinstance(provided_layers[0], Image)
    assert provided_layers[0].data.shape == shape
