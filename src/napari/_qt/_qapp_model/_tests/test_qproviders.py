"""Test app-model Qt-related providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from app_model.types import Action

from napari._app_model._app import get_app_model
from napari._qt._qapp_model.injection._qproviders import (
    _provide_active_layer,
    _provide_active_layer_list,
    _provide_qt_viewer_or_raise,
    _provide_selected_layers,
    _provide_viewer,
    _provide_viewer_or_raise,
    _provide_window_or_raise,
)
from napari._qt.qt_main_window import Window
from napari._qt.qt_viewer import QtViewer
from napari.components import LayerList
from napari.layers import Shapes
from napari.utils._proxies import PublicOnlyProxy
from napari.viewer import Viewer

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

    from napari.components import ViewerModel


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
    with pytest.raises(RuntimeError, match=r'No current `Viewer` found. test'):
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
        RuntimeError, match=r'No current `QtViewer` found. test'
    ):
        _provide_qt_viewer_or_raise(msg='test')

    # create QtViewer
    make_napari_viewer()
    viewer = _provide_qt_viewer_or_raise()
    assert isinstance(viewer, QtViewer)


def test_provide_window_or_raise(make_napari_viewer):
    """Check `_provide_window_or_raise` raises or returns `Window`."""
    # raises when no Window
    with pytest.raises(RuntimeError, match=r'No current `Window` found. test'):
        _provide_window_or_raise(msg='test')

    # create viewer (and Window)
    make_napari_viewer()
    viewer = _provide_window_or_raise()
    assert isinstance(viewer, Window)


def test_provide_active_layer(
    monkeypatch: MonkeyPatch, viewer_model: ViewerModel
):
    """Check `_provide_active_layer/_list` returns correct object."""
    monkeypatch.setattr(
        'napari._qt._qapp_model.injection._qproviders._provide_viewer',
        lambda: viewer_model,
    )

    layer_a = viewer_model.add_layer(Shapes())
    viewer_model.add_layer(Shapes())
    viewer_model.layers.selection.active = layer_a

    provided_layer = _provide_active_layer()
    assert provided_layer is layer_a

    viewer_model.layers.selection = []

    provided_layer = _provide_active_layer()
    assert provided_layer is None


def test_provide_layer_list(
    monkeypatch: MonkeyPatch, viewer_model: ViewerModel
):
    monkeypatch.setattr(
        'napari._qt._qapp_model.injection._qproviders._provide_viewer',
        lambda: viewer_model,
    )

    layer_a = viewer_model.add_layer(Shapes())
    layer_b = viewer_model.add_layer(Shapes())

    provided_layers = _provide_active_layer_list()
    assert isinstance(provided_layers, LayerList)
    assert provided_layers[0] is layer_a
    assert provided_layers[1] is layer_b


def test_provide_selected_layers(
    monkeypatch: MonkeyPatch, viewer_model: ViewerModel
) -> None:
    monkeypatch.setattr(
        'napari._qt._qapp_model.injection._qproviders._provide_viewer',
        lambda: viewer_model,
    )
    s1 = viewer_model.add_layer(Shapes())
    viewer_model.add_layer(Shapes())
    s3 = viewer_model.add_layer(Shapes())

    viewer_model.layers.selection = [s1, s3]

    selected_layers = _provide_selected_layers()
    assert selected_layers is not None
    assert len(selected_layers) == 2
    assert s1 in selected_layers
    assert s3 in selected_layers
