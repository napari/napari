import platform
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from magicgui.widgets import Container
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QImage
from qtpy.QtWidgets import QApplication, QWidget

from napari._qt.qt_main_window import QWIDGETSIZE_MAX, Window, _QtMainWindow
from napari._qt.utils import QImg2array
from napari._tests.utils import skip_on_win_ci
from napari.settings import get_settings
from napari.utils.theme import (
    _themes,
    get_theme,
    register_theme,
    unregister_theme,
)


def test_current_viewer(make_napari_viewer):
    """Test that we can retrieve the "current" viewer window easily.

    ... where "current" means it was the last viewer the user interacted with.
    """
    assert _QtMainWindow.current() is None

    # when we create a new viewer it becomes accessible at Viewer.current()
    v1 = make_napari_viewer(title='v1')
    assert _QtMainWindow._instances == [v1.window._qt_window]
    assert _QtMainWindow.current() == v1.window._qt_window

    v2 = make_napari_viewer(title='v2')
    assert _QtMainWindow._instances == [
        v1.window._qt_window,
        v2.window._qt_window,
    ]
    assert _QtMainWindow.current() == v2.window._qt_window

    # Viewer.current() will always give the most recently activated viewer.
    v1.window.activate()
    assert _QtMainWindow.current() == v1.window._qt_window
    v2.window.activate()
    assert _QtMainWindow.current() == v2.window._qt_window

    # The list remembers the z-order of previous viewers ...
    v2.close()
    assert _QtMainWindow.current() == v1.window._qt_window
    assert _QtMainWindow._instances == [v1.window._qt_window]

    # and when none are left, Viewer.current() becomes None again
    v1.close()
    assert _QtMainWindow._instances == []
    assert _QtMainWindow.current() is None


def test_set_geometry(make_napari_viewer):
    viewer = make_napari_viewer()
    values = (70, 70, 1000, 700)
    viewer.window.set_geometry(*values)
    assert viewer.window.geometry() == values


@patch.object(Window, '_update_theme_no_event')
@patch.object(Window, '_remove_theme')
@patch.object(Window, '_add_theme')
def test_update_theme(
    mock_add_theme,
    mock_remove_theme,
    mock_update_theme_no_event,
    make_napari_viewer,
):
    viewer = make_napari_viewer()

    blue = get_theme('dark')
    blue.id = 'blue'
    register_theme('blue', blue, 'test')

    # triggered when theme was added
    mock_add_theme.assert_called()
    mock_remove_theme.assert_not_called()

    unregister_theme('blue')
    # triggered when theme was removed
    mock_remove_theme.assert_called()

    mock_update_theme_no_event.assert_not_called()
    viewer.theme = 'light'
    theme = _themes['light']
    theme.icon = '#FF0000'
    mock_update_theme_no_event.assert_called()


def test_lazy_console(make_napari_viewer):
    v = make_napari_viewer()
    assert v.window._qt_viewer._console is None
    v.update_console({'test': 'test'})
    assert v.window._qt_viewer._console is None


@pytest.mark.skipif(
    platform.system() == 'Darwin', reason='Cannot control menu bar on MacOS'
)
def test_menubar_shortcut(make_napari_viewer):
    v = make_napari_viewer()
    v.show()
    assert v.window.main_menu.isVisible()
    assert not v.window._main_menu_shortcut.isEnabled()
    v.window._toggle_menubar_visible()
    assert not v.window.main_menu.isVisible()
    assert v.window._main_menu_shortcut.isEnabled()


@skip_on_win_ci
def test_screenshot_to_file(make_napari_viewer, tmp_path):
    """
    Test taking a screenshot using the Window instance and saving it to a file.
    """
    viewer = make_napari_viewer()
    screenshot_file_path = str(tmp_path / 'screenshot.png')

    np.random.seed(0)
    # Add image
    data = np.ones((10, 15), dtype=np.uint8)
    viewer.add_image(data)

    # Add labels
    data = np.random.randint(20, size=(10, 15))
    viewer.add_labels(data)

    # Add points
    data = 20 * np.random.random((10, 2))
    viewer.add_points(data)

    # Add vectors
    data = 20 * np.random.random((10, 2, 2))
    viewer.add_vectors(data)

    # Add shapes
    data = 20 * np.random.random((10, 4, 2))
    viewer.add_shapes(data)

    # Take screenshot
    screenshot_array = viewer.window.screenshot(
        screenshot_file_path, flash=False, canvas_only=True
    )
    screenshot_array_from_file = QImg2array(QImage(screenshot_file_path))
    assert np.array_equal(screenshot_array, screenshot_array_from_file)


def test_set_status_and_tooltip(make_napari_viewer):
    viewer = make_napari_viewer()
    # create active layer
    viewer.add_image(np.zeros((10, 10)))
    assert viewer.status == 'Ready'
    assert viewer.tooltip.text == ''
    viewer.window._qt_window.set_status_and_tooltip(('Text1', 'Text2'))
    assert viewer.status == 'Text1'
    assert viewer.tooltip.text == 'Text2'
    viewer.window._qt_window.set_status_and_tooltip(None)
    assert viewer.status == 'Text1'
    assert viewer.tooltip.text == 'Text2'


@pytest.mark.parametrize('BaseClass', [Container, QWidget])
def test_add_plugin_dock_widget(make_napari_viewer, monkeypatch, BaseClass):
    """Test that we can add a plugin dock widget to the viewer."""

    class InnerWidget(BaseClass):
        pass

    mock = MagicMock(return_value=(InnerWidget, 'widget name'))
    monkeypatch.setattr('napari.plugins._npe2.get_widget_contribution', mock)
    viewer = make_napari_viewer()
    assert list(viewer.window.dock_widgets.keys()) == []

    docked, widget = viewer.window.add_plugin_dock_widget(
        'sample_plugin', 'sample_widget'
    )
    assert isinstance(widget, InnerWidget)
    assert docked.inner_widget() is widget
    assert list(viewer.window.dock_widgets.keys()) == [
        'widget name (sample_plugin)'
    ]
    assert viewer.window.dock_widgets['widget name (sample_plugin)'] is widget
    docked2, widget2 = viewer.window.add_plugin_dock_widget(
        'sample_plugin', 'sample_widget'
    )
    assert docked is docked2
    assert widget is widget2

    with pytest.raises(TypeError):
        viewer.window.dock_widgets['widget name (sample_plugin)'] = 1


def test_sliding_dock_area(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    settings = get_settings()

    assert not settings.appearance.dock_area_autohide
    assert not qt_window.widgets_sliding_dock_area

    settings.appearance.dock_area_autohide = True
    assert all(
        not dock.isVisible() for dock in qt_window.widgets_sliding_dock_area
    )
    assert len(qt_window.widgets_sliding_dock_area) == 2

    for dock in qt_window.widgets_sliding_dock_area:
        assert dock.maximumWidth() != QWIDGETSIZE_MAX or (
            dock.maximumHeight() != QWIDGETSIZE_MAX
        )

    get_settings().appearance.dock_area_autohide = False
    assert not qt_window.widgets_sliding_dock_area
    assert all(
        dock.isVisible()
        for dock in [
            viewer.window._qt_viewer.dockLayerControls,
            viewer.window._qt_viewer.dockLayerList,
        ]
    )

    settings.appearance.dock_area_autohide = True
    assert len(qt_window.widgets_sliding_dock_area) == 2
    assert all(
        not dock.isVisible() for dock in qt_window.widgets_sliding_dock_area
    )
    viewer.close()


def test_sliding_dock_area_disable_restores_user_size(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    settings = get_settings()
    settings.appearance.dock_area_autohide = True

    dock = viewer.window._qt_viewer.dockLayerList
    # This will act like a resize actually happened
    qt_window.widgets_sliding_dock_area[dock]['user_size'] = 444

    settings.appearance.dock_area_autohide = False
    QApplication.processEvents()
    assert dock.width() == 444

    for state_dict in qt_window.widgets_sliding_dock_area.values():
        state_dict['animation'].stop()
    viewer.close()


def test_sliding_dock_redock_to_different_area_resets_state(
    make_napari_viewer,
):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    settings = get_settings()
    settings.appearance.dock_area_autohide = True

    dock = viewer.window._qt_viewer.dockLayerList
    original_area = qt_window.dockWidgetArea(dock)
    new_area = (
        Qt.DockWidgetArea.RightDockWidgetArea
        if original_area == Qt.DockWidgetArea.LeftDockWidgetArea
        else Qt.DockWidgetArea.LeftDockWidgetArea
    )

    # expand it and give it manual sizes on its original side
    qt_window._slide_dock_generic(
        dock, show=True, property_name=b'maximumWidth'
    )
    qt_window.widgets_sliding_dock_area[dock]['animation'].stop()
    qt_window.widgets_sliding_dock_area[dock]['user_size'] = 555
    qt_window.widgets_sliding_dock_area[dock]['cross_axis_size'] = 321

    dock.setFloating(True)
    assert dock not in qt_window.widgets_sliding_dock_area

    qt_window.addDockWidget(new_area, dock)
    dock.setFloating(False)

    assert dock in qt_window.widgets_sliding_dock_area
    state = qt_window.widgets_sliding_dock_area[dock]
    assert qt_window.dockWidgetArea(dock) == new_area
    assert state['user_size'] is None
    assert state['cross_axis_size'] is None
    assert state['visible_state'] is False
    assert state['animation'] is None

    viewer.close()


def test_sliding_dock_cross_axis_size_preserved_across_cycles(
    make_napari_viewer,
):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    settings = get_settings()
    settings.appearance.dock_area_autohide = True

    dock_a = viewer.window._qt_viewer.dockLayerControls
    dock_b = viewer.window._qt_viewer.dockLayerList
    assert qt_window.dockWidgetArea(dock_a) == qt_window.dockWidgetArea(dock_b)

    # Need these functions as the sliding happens on the dock area and not single widget
    def expand_both():
        for dock in (dock_a, dock_b):
            qt_window._slide_dock_generic(
                dock, show=True, property_name=b'maximumWidth'
            )
            qt_window.widgets_sliding_dock_area[dock]['animation'].stop()
            qt_window._on_dock_size_animated(
                dock, dock.width(), Qt.Orientation.Horizontal
            )
        QApplication.processEvents()

    def collapse_both():
        for dock in (dock_a, dock_b):
            qt_window._slide_dock_generic(
                dock, show=False, property_name=b'maximumWidth'
            )
            qt_window.widgets_sliding_dock_area[dock]['animation'].stop()
        QApplication.processEvents()

    # Give a remembered height for dock b when expanded. This is a request, can be that it is not exactly 123
    # We do a sanity check that at least the height indeed changed.
    expand_both()
    height_before_resize = dock_b.height()
    qt_window.resizeDocks([dock_b], [123], Qt.Orientation.Vertical)
    QApplication.processEvents()
    assert dock_b.height() != height_before_resize

    qt_window.widgets_sliding_dock_area[dock_b]['cross_axis_size'] = None
    qt_window._on_dock_size_animated(
        dock_b, dock_b.width(), Qt.Orientation.Horizontal
    )
    QApplication.processEvents()
    remembered_height = qt_window.widgets_sliding_dock_area[dock_b][
        'cross_axis_size'
    ]
    assert remembered_height == dock_b.height()

    for _ in range(3):
        collapse_both()
        expand_both()
        assert dock_b.height() == remembered_height

    for state_dict in qt_window.widgets_sliding_dock_area.values():
        state_dict['animation'].stop()
    viewer.close()


def test_hover_at_left_edge_expands_dock(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    settings = get_settings()
    settings.appearance.dock_area_autohide = True

    dock = viewer.window._qt_viewer.dockLayerList
    assert not dock.isVisible()

    qt_window._handle_multi_dock_hover(QPoint(0, qt_window.height() // 2))

    assert qt_window.widgets_sliding_dock_area[dock]['visible_state'] is True
    assert dock.isVisible() is True

    for state_dict in qt_window.widgets_sliding_dock_area.values():
        state_dict['animation'].stop()
    viewer.close()


def test_hover_away_from_center_does_not_expand_dock(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    settings = get_settings()
    settings.appearance.dock_area_autohide = True

    dock = viewer.window._qt_viewer.dockLayerList

    qt_window._handle_multi_dock_hover(
        QPoint(qt_window.width() // 2, qt_window.height() // 2)
    )

    assert qt_window.widgets_sliding_dock_area[dock]['visible_state'] is False
    assert not dock.isVisible()

    viewer.close()


def test_hover_away_from_expanded_dock_collapses_it(make_napari_viewer, qtbot):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    settings = get_settings()
    settings.appearance.dock_area_autohide = True

    dock = viewer.window._qt_viewer.dockLayerList

    qt_window._handle_multi_dock_hover(QPoint(0, qt_window.height() // 2))
    assert qt_window.widgets_sliding_dock_area[dock]['visible_state'] is True

    qt_window._handle_multi_dock_hover(
        QPoint(qt_window.width() - 1, qt_window.height() // 2)
    )

    assert qt_window.widgets_sliding_dock_area[dock]['visible_state'] is False

    for state_dict in qt_window.widgets_sliding_dock_area.values():
        state_dict['animation'].stop()
    viewer.close()


def test_hover_does_not_affect_dock_when_autohide_disabled(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    settings = get_settings()
    assert not settings.appearance.dock_area_autohide

    dock = viewer.window._qt_viewer.dockLayerList
    assert dock not in qt_window.widgets_sliding_dock_area
    was_visible = dock.isVisible()

    qt_window._handle_multi_dock_hover(QPoint(0, qt_window.height() // 2))

    assert dock.isVisible() == was_visible
