import os
import sys

import numpy as np
import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QApplication

from napari._app_model import get_app_model
from napari._qt._qapp_model.qactions._view import (
    _get_current_tooltip_visibility,
    toggle_action_details,
)
from napari._tests.utils import skip_local_focus, skip_local_popups


def check_windows_style(viewer):
    if os.name != 'nt':
        return
    import win32con
    import win32gui

    window_handle = viewer.window._qt_window.windowHandle()
    window_handle_id = int(window_handle.winId())
    window_style = win32gui.GetWindowLong(window_handle_id, win32con.GWL_STYLE)
    assert window_style & win32con.WS_BORDER == win32con.WS_BORDER


def check_view_menu_visibility(viewer, qtbot):
    if viewer.window._qt_window.menuBar().isNativeMenuBar():
        return

    assert not viewer.window.view_menu.isVisible()
    qtbot.keyClick(
        viewer.window._qt_window.menuBar(), Qt.Key_V, modifier=Qt.AltModifier
    )
    qtbot.waitUntil(viewer.window.view_menu.isVisible)
    viewer.window.view_menu.close()
    assert not viewer.window.view_menu.isVisible()


@pytest.mark.parametrize(
    ('action_id', 'action_title', 'viewer_attr', 'sub_attr'),
    toggle_action_details,
)
def test_toggle_axes_scale_bar_attr(
    make_napari_viewer, action_id, action_title, viewer_attr, sub_attr
):
    """
    Test toggle actions related with viewer axes and scale bar attributes.

    * Viewer `axes` attributes:
        * `visible`
        * `colored`
        * `labels`
        * `dashed`
        * `arrows`
    * Viewer `scale_bar` attributes:
        * `visible`
        * `colored`
        * `ticks`
    """
    app = get_app_model()
    viewer = make_napari_viewer()

    # Get viewer attribute to check (`axes` or `scale_bar`)
    axes_scale_bar = getattr(viewer, viewer_attr)

    # Get initial sub-attribute value (for example `axes.visible`)
    initial_value = getattr(axes_scale_bar, sub_attr)

    # Change sub-attribute via action command execution and check value
    app.commands.execute_command(action_id)
    changed_value = getattr(axes_scale_bar, sub_attr)
    assert initial_value is not changed_value


@skip_local_popups
@pytest.mark.qt_log_level_fail('WARNING')
def test_toggle_fullscreen_from_normal(make_napari_viewer, qtbot):
    """
    Test toggle fullscreen action from normal window state.

    Check that toggling from a normal state can be done without
    generating any type of warning and menu bar elements are visible in any
    window state.
    """
    action_id = 'napari.window.view.toggle_fullscreen'
    app = get_app_model()
    viewer = make_napari_viewer(show=True)

    # Check initial default state (no fullscreen)
    assert not viewer.window._qt_window.isFullScreen()

    # Check `View` menu can be seen in normal window state
    check_view_menu_visibility(viewer, qtbot)

    # Check fullscreen state change
    app.commands.execute_command(action_id)
    if sys.platform == 'darwin':
        # On macOS, wait for the animation to complete
        qtbot.wait(250)
    assert viewer.window._qt_window.isFullScreen()
    check_windows_style(viewer)

    # Check `View` menu can be seen in fullscreen window state
    check_view_menu_visibility(viewer, qtbot)

    # Check return to non fullscreen state
    app.commands.execute_command(action_id)
    if sys.platform == 'darwin':
        # On macOS, wait for the animation to complete
        qtbot.wait(250)
    assert not viewer.window._qt_window.isFullScreen()
    check_windows_style(viewer)

    # Check `View` still menu can be seen in non fullscreen window state
    check_view_menu_visibility(viewer, qtbot)


@skip_local_popups
@pytest.mark.qt_log_level_fail('WARNING')
def test_toggle_fullscreen_from_maximized(make_napari_viewer, qtbot):
    """
    Test toggle fullscreen action from maximized window state.

    Check that toggling from a maximized state can be done without
    generating any type of warning and menu bar elements are visible in any
    window state.
    """
    action_id = 'napari.window.view.toggle_fullscreen'
    app = get_app_model()
    viewer = make_napari_viewer(show=True)

    # Check fullscreen state change while maximized
    assert not viewer.window._qt_window.isMaximized()
    viewer.window._qt_window.showMaximized()

    # Check `View` menu can be seen in maximized window state
    check_view_menu_visibility(viewer, qtbot)

    # Check fullscreen state change
    app.commands.execute_command(action_id)
    if sys.platform == 'darwin':
        # On macOS, wait for the animation to complete
        qtbot.wait(250)
    assert viewer.window._qt_window.isFullScreen()
    check_windows_style(viewer)

    # Check `View` menu can be seen in fullscreen window state coming from maximized state
    check_view_menu_visibility(viewer, qtbot)

    # Check return to non fullscreen state
    app.commands.execute_command(action_id)
    if sys.platform == 'darwin':
        # On macOS, wait for the animation to complete
        qtbot.wait(250)

    def check_not_fullscreen():
        assert not viewer.window._qt_window.isFullScreen()

    qtbot.waitUntil(check_not_fullscreen)
    check_windows_style(viewer)

    # Check `View` still menu can be seen in non fullscreen window state
    check_view_menu_visibility(viewer, qtbot)


@skip_local_focus
@pytest.mark.skipif(
    sys.platform == 'darwin',
    reason='Toggle menubar action not enabled on macOS',
)
def test_toggle_menubar(make_napari_viewer, qtbot):
    """
    Test menubar toggle functionality.

    Skipped on macOS since the menubar is the system one so the menubar
    toggle action doesn't exist/isn't enabled there.
    """
    action_id = 'napari.window.view.toggle_menubar'
    app = get_app_model()
    viewer = make_napari_viewer(show=True)

    # Check initial state (visible menubar)
    assert viewer.window._qt_window.menuBar().isVisible()
    assert not viewer.window._qt_window._toggle_menubar_visibility

    # Check menubar gets hidden
    app.commands.execute_command(action_id)
    assert not viewer.window._qt_window.menuBar().isVisible()
    assert viewer.window._qt_window._toggle_menubar_visibility

    # Check menubar gets visible via mouse hovering over the window top area
    qtbot.mouseMove(viewer.window._qt_window, pos=QPoint(10, 10))
    qtbot.mouseMove(viewer.window._qt_window, pos=QPoint(15, 15))
    qtbot.waitUntil(viewer.window._qt_window.menuBar().isVisible)

    # Check menubar hides when the mouse no longer is hovering over the window top area
    qtbot.mouseMove(viewer.window._qt_window, pos=QPoint(50, 50))
    qtbot.waitUntil(lambda: not viewer.window._qt_window.menuBar().isVisible())

    # Check restore menubar visibility
    app.commands.execute_command(action_id)
    assert viewer.window._qt_window.menuBar().isVisible()
    assert not viewer.window._qt_window._toggle_menubar_visibility


def test_toggle_play(make_napari_viewer, qtbot):
    """Test toggle play action."""
    action_id = 'napari.window.view.toggle_play'
    app = get_app_model()
    viewer = make_napari_viewer()

    # Check action on empty viewer
    with pytest.warns(
        expected_warning=UserWarning, match='Refusing to play a hidden axis'
    ):
        app.commands.execute_command(action_id)

    # Check action on viewer with layer
    np.random.seed(0)
    data = np.random.random((10, 10, 15))
    viewer.add_image(data)
    # Assert action triggers play
    app.commands.execute_command(action_id)
    qtbot.waitUntil(lambda: viewer.window._qt_viewer.dims.is_playing)
    # Assert action stops play
    with qtbot.waitSignal(
        viewer.window._qt_viewer.dims._animation_thread.finished
    ):
        app.commands.execute_command(action_id)
        QApplication.processEvents()
    qtbot.waitUntil(lambda: not viewer.window._qt_viewer.dims.is_playing)


@skip_local_popups
def test_toggle_activity_dock(make_napari_viewer):
    """Test toggle activity dock"""
    action_id = 'napari.window.view.toggle_activity_dock'
    app = get_app_model()
    viewer = make_napari_viewer(show=True)

    # Check initial activity dock state (hidden)
    assert not viewer.window._qt_window._activity_dialog.isVisible()
    assert (
        viewer.window._status_bar._activity_item._activityBtn.arrowType()
        == Qt.ArrowType.UpArrow
    )

    # Check activity dock gets visible
    app.commands.execute_command(action_id)
    assert viewer.window._qt_window._activity_dialog.isVisible()
    assert (
        viewer.window._status_bar._activity_item._activityBtn.arrowType()
        == Qt.ArrowType.DownArrow
    )

    # Restore activity dock visibility (hidden)
    app.commands.execute_command(action_id)
    assert not viewer.window._qt_window._activity_dialog.isVisible()
    assert (
        viewer.window._status_bar._activity_item._activityBtn.arrowType()
        == Qt.ArrowType.UpArrow
    )


def test_toggle_layer_tooltips(make_napari_viewer, qtbot):
    """Test toggle layer tooltips"""
    make_napari_viewer()
    action_id = 'napari.window.view.toggle_layer_tooltips'
    app = get_app_model()

    # Check initial layer tooltip visibility settings state (False)
    assert not _get_current_tooltip_visibility()

    # Check layer tooltip visibility toggle
    app.commands.execute_command(action_id)
    assert _get_current_tooltip_visibility()

    # Restore layer tooltip visibility
    app.commands.execute_command(action_id)
    assert not _get_current_tooltip_visibility()


def test_zoom_actions(make_napari_viewer):
    """Test zoom actions"""
    viewer = make_napari_viewer()
    app = get_app_model()

    viewer.add_image(np.ones((10, 10, 10)))

    # get initial zoom state
    initial_zoom = viewer.camera.zoom

    # Check zoom in action
    app.commands.execute_command('napari.viewer.camera.zoom_in')
    assert viewer.camera.zoom == pytest.approx(1.5 * initial_zoom)

    # Check zoom out action
    app.commands.execute_command('napari.viewer.camera.zoom_out')
    assert viewer.camera.zoom == pytest.approx(initial_zoom)

    viewer.camera.zoom = 2
    # Check reset zoom action
    app.commands.execute_command('napari.viewer.fit_to_view')
    assert viewer.camera.zoom == pytest.approx(initial_zoom)

    # Check that angle is preserved
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (90, 0, 0)
    viewer.camera.zoom = 2
    app.commands.execute_command('napari.viewer.fit_to_view')
    # Zoom should be reset, but angle unchanged
    assert viewer.camera.zoom == pytest.approx(initial_zoom)
    assert viewer.camera.angles == (90, 0, 0)
