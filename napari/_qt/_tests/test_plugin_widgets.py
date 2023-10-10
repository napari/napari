from unittest.mock import patch

import pytest
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget

import napari
from napari import Viewer
from napari._qt.qt_main_window import _instantiate_dock_widget
from napari.utils._proxies import PublicOnlyProxy


class Widg1(QWidget):
    pass


class Widg2(QWidget):
    def __init__(self, napari_viewer) -> None:
        self.viewer = napari_viewer
        super().__init__()


class Widg3(QWidget):
    def __init__(self, v: Viewer) -> None:
        self.viewer = v
        super().__init__()

    def fail(self):
        """private attr not allowed"""
        self.viewer.window._qt_window


def magicfunc(viewer: 'napari.Viewer'):
    return viewer


dwidget_args = {
    'single_class': Widg1,
    'class_tuple': (Widg1, {'area': 'right'}),
    'tuple_list': [(Widg1, {'area': 'right'}), (Widg2, {})],
    'tuple_list2': [(Widg1, {'area': 'right'}), Widg2],
    'bad_class': 1,
    'bad_tuple1': (Widg1, 1),
    'bad_double_tuple': ((Widg1, {}), (Widg2, {})),
}


# napari_plugin_manager from _testsupport.py
# monkeypatch, request, recwarn fixtures are from pytest
@pytest.mark.parametrize('arg', dwidget_args.values(), ids=dwidget_args.keys())
def test_dock_widget_registration(
    arg, napari_plugin_manager, request, recwarn
):
    """Test that dock widgets get validated and registerd correctly."""

    class Plugin:
        @napari_hook_implementation
        def napari_experimental_provide_dock_widget():
            return arg

    napari_plugin_manager.register(Plugin, name='Plugin')
    napari_plugin_manager.discover_widgets()
    widgets = napari_plugin_manager._dock_widgets

    if '[bad_' in request.node.name:
        assert len(recwarn) == 1
        assert not widgets
    else:
        assert len(recwarn) == 0
        assert widgets['Plugin']['Widg1'][0] == Widg1
        if 'tuple_list' in request.node.name:
            assert widgets['Plugin']['Widg2'][0] == Widg2


@pytest.fixture
def test_plugin_widgets(monkeypatch, napari_plugin_manager):
    """A smattering of example registered dock widgets and function widgets."""
    tnpm = napari_plugin_manager
    dock_widgets = {
        "TestP1": {"Widg1": (Widg1, {}), "Widg2": (Widg2, {})},
        "TestP2": {"Widg3": (Widg3, {})},
    }
    monkeypatch.setattr(tnpm, "_dock_widgets", dock_widgets)

    function_widgets = {'TestP3': {'magic': magicfunc}}
    monkeypatch.setattr(tnpm, "_function_widgets", function_widgets)
    yield


def test_inject_viewer_proxy(make_napari_viewer):
    """Test that the injected viewer is a public-only proxy"""
    viewer = make_napari_viewer()
    wdg = _instantiate_dock_widget(Widg3, viewer)
    assert isinstance(wdg.viewer, PublicOnlyProxy)

    # simulate access from outside napari
    with patch('napari.utils.misc.ROOT_DIR', new='/some/other/package'):
        with pytest.warns(FutureWarning):
            wdg.fail()
