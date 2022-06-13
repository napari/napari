from unittest.mock import patch

import pytest
from qtpy.QtWidgets import QWidget

import napari
from napari import Viewer
from napari._qt.qt_main_window import _instantiate_dock_widget
from napari.utils._proxies import PublicOnlyProxy


class Widg1(QWidget):
    pass


class Widg2(QWidget):
    def __init__(self, napari_viewer):
        self.viewer = napari_viewer
        super().__init__()


class Widg3(QWidget):
    def __init__(self, v: Viewer):
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


def test_plugin_widgets_menus(test_plugin_widgets, make_napari_viewer):
    """Test the plugin widgets get added to the window menu correctly."""
    viewer = make_napari_viewer()
    # only take the plugin actions
    actions = viewer.window.plugins_menu.actions()
    for cnt, action in enumerate(actions):
        if action.text() == "":
            # this is the separator
            break
    actions = actions[cnt + 1 :]
    texts = [a.text() for a in actions]
    for t in ['TestP1', 'TestP2: Widg3', 'TestP3: magic']:
        assert t in texts

    # Expect a submenu ("Test plugin1") with particular entries.
    tp1 = next(m for m in actions if m.text() == 'TestP1')
    assert tp1.menu()
    assert [a.text() for a in tp1.menu().actions()] == ['Widg1', 'Widg2']


def test_making_plugin_dock_widgets(test_plugin_widgets, make_napari_viewer):
    """Test that we can create dock widgets, and they get the viewer."""
    viewer = make_napari_viewer()
    # only take the plugin actions
    actions = viewer.window.plugins_menu.actions()
    for cnt, action in enumerate(actions):
        if action.text() == "":
            # this is the separator
            break
    actions = actions[cnt + 1 :]

    # trigger the 'TestP2: Widg3' action
    tp2 = next(m for m in actions if m.text().startswith('TestP2'))
    tp2.trigger()
    # make sure that a dock widget was created
    assert 'TestP2: Widg3' in viewer.window._dock_widgets
    dw = viewer.window._dock_widgets['TestP2: Widg3']
    assert isinstance(dw.widget(), Widg3)
    # This widget uses the parameter annotation method to receive a viewer
    assert isinstance(dw.widget().viewer, napari.Viewer)
    # Add twice is ok, only does a show
    tp2.trigger()

    # trigger the 'TestP1 > Widg2' action (it's in a submenu)
    tp2 = next(m for m in actions if m.text().startswith('TestP1'))
    action = tp2.menu().actions()[1]
    assert action.text() == 'Widg2'
    action.trigger()
    # make sure that a dock widget was created
    assert 'TestP1: Widg2' in viewer.window._dock_widgets
    dw = viewer.window._dock_widgets['TestP1: Widg2']
    assert isinstance(dw.widget(), Widg2)
    # This widget uses parameter *name* "napari_viewer" to get a viewer
    assert isinstance(dw.widget().viewer, napari.Viewer)
    # Add twice is ok, only does a show
    action.trigger()
    # Check that widget is still there when closed.
    widg = dw.widget()
    dw.title.hide_button.click()
    assert widg
    # Check that widget is destroyed when closed.
    dw.destroyOnClose()
    assert action not in viewer.window.plugins_menu.actions()
    assert not widg.parent()


def test_making_function_dock_widgets(test_plugin_widgets, make_napari_viewer):
    """Test that we can create magicgui widgets, and they get the viewer."""
    import magicgui

    viewer = make_napari_viewer()
    # only take the plugin actions
    actions = viewer.window.plugins_menu.actions()
    for cnt, action in enumerate(actions):
        if action.text() == "":
            # this is the separator
            break
    actions = actions[cnt + 1 :]

    # trigger the 'TestP3: magic' action
    tp3 = next(m for m in actions if m.text().startswith('TestP3'))
    tp3.trigger()
    # make sure that a dock widget was created
    assert 'TestP3: magic' in viewer.window._dock_widgets
    dw = viewer.window._dock_widgets['TestP3: magic']
    # make sure that it contains a magicgui widget
    magic_widget = dw.widget()._magic_widget
    FGui = getattr(magicgui.widgets, 'FunctionGui', None)
    if FGui is None:
        # pre magicgui 0.2.6
        FGui = magicgui.FunctionGui
    assert isinstance(magic_widget, FGui)
    # This magicgui widget uses the parameter annotation to receive a viewer
    assert isinstance(magic_widget.viewer.value, napari.Viewer)
    # The function just returns the viewer... make sure we can call it
    assert isinstance(magic_widget(), napari.Viewer)
    # Add twice is ok, only does a show
    tp3.trigger()


def test_inject_viewer_proxy(make_napari_viewer):
    """Test that the injected viewer is a public-only proxy"""
    viewer = make_napari_viewer()
    wdg = _instantiate_dock_widget(Widg3, viewer)
    assert isinstance(wdg.viewer, PublicOnlyProxy)

    # simulate access from outside napari
    with patch('napari.utils.misc.ROOT_DIR', new='/some/other/package'):
        with pytest.warns(FutureWarning):
            wdg.fail()
