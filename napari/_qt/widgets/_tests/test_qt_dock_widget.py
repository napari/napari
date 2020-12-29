import pytest
from qtpy.QtWidgets import QDockWidget, QHBoxLayout, QPushButton, QVBoxLayout


def test_add_dock_widget(make_test_viewer):
    """Test basic add_dock_widget functionality"""
    viewer = make_test_viewer()
    widg = QPushButton('button')
    dwidg = viewer.window.add_dock_widget(widg, name='test')
    assert not dwidg.is_vertical
    assert viewer.window._qt_window.findChild(QDockWidget, 'test')
    assert dwidg.widget() == widg
    dwidg._on_visibility_changed(True)  # smoke test

    widg2 = QPushButton('button')
    dwidg2 = viewer.window.add_dock_widget(widg2, name='test2', area='right')
    assert dwidg2.is_vertical
    assert viewer.window._qt_window.findChild(QDockWidget, 'test2')
    assert dwidg2.widget() == widg2
    dwidg2._on_visibility_changed(True)  # smoke test

    with pytest.raises(ValueError):
        # 'under' is not a valid area
        viewer.window.add_dock_widget(widg2, name='test2', area='under')

    with pytest.raises(ValueError):
        # 'under' is not a valid area
        viewer.window.add_dock_widget(
            widg2, name='test2', allowed_areas=['under']
        )

    with pytest.raises(TypeError):
        # allowed_areas must be a list
        viewer.window.add_dock_widget(
            widg2, name='test2', allowed_areas='under'
        )


def test_add_dock_widget_from_list(make_test_viewer):
    """Test that we can add a list of widgets and they will be combined"""
    viewer = make_test_viewer()
    widg = QPushButton('button')
    widg2 = QPushButton('button')

    dwidg = viewer.window.add_dock_widget(
        [widg, widg2], name='test', area='right'
    )
    assert viewer.window._qt_window.findChild(QDockWidget, 'test')
    assert isinstance(dwidg.widget().layout, QVBoxLayout)

    dwidg = viewer.window.add_dock_widget(
        [widg, widg2], name='test2', area='bottom'
    )
    assert viewer.window._qt_window.findChild(QDockWidget, 'test2')
    assert isinstance(dwidg.widget().layout, QHBoxLayout)


def test_add_dock_widget_raises(make_test_viewer):
    """Test that the widget passed must be a DockWidget."""
    viewer = make_test_viewer()
    widg = object()

    with pytest.raises(TypeError):
        viewer.window.add_dock_widget(widg, name='test')


def test_remove_dock_widget_orphans_widget(make_test_viewer):
    viewer = make_test_viewer()
    widg = QPushButton('button')

    assert not widg.parent()
    dw = viewer.window.add_dock_widget(widg, name='test')
    assert widg.parent() is dw
    assert dw.toggleViewAction() in viewer.window.window_menu.actions()
    viewer.window.remove_dock_widget(dw)
    assert dw.toggleViewAction() not in viewer.window.window_menu.actions()
    del dw
    # if dw didn't release widg, we'd get an exception when next accessing widg
    assert not widg.parent()


def test_remove_dock_widget_by_widget_reference(make_test_viewer):
    viewer = make_test_viewer()
    widg = QPushButton('button')

    dw = viewer.window.add_dock_widget(widg, name='test')
    assert widg.parent() is dw
    assert dw in viewer.window._qt_window.findChildren(QDockWidget)
    viewer.window.remove_dock_widget(widg)
    with pytest.raises(LookupError):
        # it's gone this time:
        viewer.window.remove_dock_widget(widg)
    assert not widg.parent()
