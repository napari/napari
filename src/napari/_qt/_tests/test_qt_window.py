import platform
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from magicgui.widgets import Container
from qtpy.QtGui import QImage
from qtpy.QtWidgets import QWidget

from napari._qt.qt_main_window import (
    Window,
    _QtMainWindow,
    _shutdown_open_windows,
)
from napari._qt.utils import QImg2array
from napari._tests.utils import skip_on_win_ci
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


def test_shutdown_stops_worker_threads(make_napari_viewer):
    viewer = make_napari_viewer()
    window = viewer.window._qt_window
    assert window in _QtMainWindow._instances

    with (
        patch.object(window.status_thread, 'close_terminate') as stop_status,
        patch.object(window._qt_viewer.dims, 'stop') as stop_animation,
    ):
        _shutdown_open_windows()

    stop_status.assert_called_once_with()
    stop_animation.assert_called_once_with()


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


def test_status_tooltip(make_napari_viewer):
    viewer = make_napari_viewer()
    assert viewer.window._qt_window._current_tooltip == 'Ready'
    viewer.add_points()
    viewer.cursor.canvas_position = (10, 10)
    # cannot rely on statuschecker cause it's disabled in make_napari_viewer
    # so we need to trigger ourselves
    assert viewer._calc_status_from_cursor() != 'Ready'


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
