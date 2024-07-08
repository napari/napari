import sys

import numpy as np
import pytest
from qtpy.QtCore import QPoint, Qt

from napari._app_model import get_app
from napari._qt._qapp_model.qactions._view import (
    _get_current_tooltip_visibility,
    toggle_action_details,
)
from napari._tests.utils import skip_local_focus, skip_local_popups


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
    app = get_app()
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
def test_toggle_fullscreen(make_napari_viewer, qtbot):
    """Test toggle fullscreen action."""
    action_id = 'napari.window.view.toggle_fullscreen'
    app = get_app()
    viewer = make_napari_viewer(show=True)

    # Check initial default state (no fullscreen)
    assert not viewer.window._qt_window.isFullScreen()

    # Check fullscreen state change
    app.commands.execute_command(action_id)
    if sys.platform == 'darwin':
        # On macOS, wait for the animation to complete
        qtbot.wait(250)
    assert viewer.window._qt_window.isFullScreen()

    # Check return to non fullscreen state
    app.commands.execute_command(action_id)
    if sys.platform == 'darwin':
        # On macOS, wait for the animation to complete
        qtbot.wait(250)
    assert not viewer.window._qt_window.isFullScreen()


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
    app = get_app()
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
    app = get_app()
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
    app.commands.execute_command(action_id)
    qtbot.waitUntil(lambda: not viewer.window._qt_viewer.dims.is_playing)


@skip_local_popups
def test_toggle_activity_dock(make_napari_viewer):
    """Test toggle activity dock"""
    action_id = 'napari.window.view.toggle_activity_dock'
    app = get_app()
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
    app = get_app()

    # Check initial layer tooltip visibility settings state (False)
    assert not _get_current_tooltip_visibility()

    # Check layer tooltip visibility toggle
    app.commands.execute_command(action_id)
    assert _get_current_tooltip_visibility()

    # Restore layer tooltip visibility
    app.commands.execute_command(action_id)
    assert not _get_current_tooltip_visibility()
