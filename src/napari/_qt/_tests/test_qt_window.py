import platform
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from magicgui.widgets import Container
from qtpy.QtCore import QEvent, QPoint
from qtpy.QtGui import QHelpEvent, QImage
from qtpy.QtWidgets import QToolTip, QWidget

from napari._qt.qt_main_window import (
    Window,
    _QtMainWindow,
    _shutdown_open_windows,
)
from napari._qt.utils import QImg2array
from napari._qt.widgets.qt_viewer_status_bar import (
    _EXPERIMENTAL_FEATURES_TO_WARN,
)
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

    bar = viewer.window._qt_window.statusBar()
    for name in _EXPERIMENTAL_FEATURES_TO_WARN:
        assert name in bar._warn_labels


@patch.object(QToolTip, 'showText')
def test_viewer_tooltip_help_event(show_text, make_napari_viewer):
    """ToolTip events are QHelpEvents, which have globalPos() only."""
    viewer = make_napari_viewer()
    viewer.tooltip.visible = True
    viewer.tooltip.text = 'Tooltip text'
    qt_window = viewer.window._qt_window
    event = QHelpEvent(QEvent.Type.ToolTip, QPoint(5, 5), QPoint(105, 205))

    qt_window.event(event)

    show_text.assert_called_once()
    assert show_text.call_args[0][0] == QPoint(105, 205)
    assert show_text.call_args[0][1] == 'Tooltip text'


@pytest.mark.parametrize('BaseClass', [Container, QWidget])
def test_add_plugin_dock_widget(make_napari_viewer, monkeypatch, BaseClass):
    """Test that we can add a plugin dock widget to the viewer."""

    class InnerWidget(BaseClass):
        pass

    mock = MagicMock(return_value=(InnerWidget, 'widget name', 'bottom', True))
    monkeypatch.setattr('napari.plugins._npe2.get_widget_contribution', mock)
    viewer = make_napari_viewer()
    assert list(viewer.window.dock_widgets.keys()) == []

    docked, widget = viewer.window.add_plugin_dock_widget(
        'sample_plugin', 'sample_widget'
    )
    assert docked.area == 'bottom'
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

    # the saved area is looked up by the widget's full name
    positions = get_settings().application.plugin_widget_positions
    positions['moved widget (sample_plugin)'] = 'left'
    mock.return_value = (InnerWidget, 'moved widget', 'bottom', False)
    docked, _ = viewer.window.add_plugin_dock_widget(
        'sample_plugin', 'moved widget'
    )
    assert docked.area == 'left'


@pytest.mark.parametrize(
    ('saved_area', 'requested_area', 'default_area', 'expected_area'),
    [
        # never moved by the user, no area passed: the widget's default
        # (e.g. a plugin's `default_dock_area`)
        (None, None, 'bottom', 'bottom'),
        # ... and with no default either, 'right'
        (None, None, None, 'right'),
        # user moved it to the left in an earlier session: their choice wins
        ('left', None, 'bottom', 'left'),
        # area passed in code (e.g. `add_dock_widget(..., area=...)`) wins
        ('left', 'top', 'bottom', 'top'),
    ],
    ids=['default', 'fallback', 'saved-beats-default', 'explicit-wins'],
)
def test_resolve_dock_area(
    saved_area, requested_area, default_area, expected_area
):
    if saved_area is not None:
        positions = get_settings().application.plugin_widget_positions
        positions['test'] = saved_area
    area = Window._resolve_dock_area('test', requested_area, default_area)
    assert area == expected_area


@pytest.mark.parametrize(
    ('requested_tabify', 'default_tabify', 'expected_tabify'),
    [
        # no tabify passed: the widget's default (e.g. a plugin's `tabify`)
        (None, True, True),
        (None, False, False),
        # tabify passed in code wins over the default
        (False, True, False),
        (True, False, True),
    ],
    ids=['default-true', 'default-false', 'explicit-false', 'explicit-true'],
)
def test_resolve_tabify(requested_tabify, default_tabify, expected_tabify):
    tabify = Window._resolve_tabify(requested_tabify, default_tabify)
    assert tabify is expected_tabify
