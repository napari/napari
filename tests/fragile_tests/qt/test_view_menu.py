import sys

import pytest
from PyQt6.QtCore import QPoint

from napari._app_model import get_app_model
from napari._tests.utils import skip_local_focus


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
    viewer.window._qt_window.move(0, 0)
    qtbot.waitUntil(viewer.window._qt_window.isVisible)
    # Check menubar gets visible via mouse hovering over the window top area
    qtbot.mouseMove(viewer.window._qt_window)
    qtbot.wait(50)
    qtbot.mouseMove(viewer.window._qt_window, pos=QPoint(15, 15))
    qtbot.waitUntil(viewer.window._qt_window.menuBar().isVisible)

    # Check menubar hides when the mouse no longer is hovering over the window top area
    qtbot.mouseMove(viewer.window._qt_window, pos=QPoint(50, 50))
    qtbot.waitUntil(lambda: not viewer.window._qt_window.menuBar().isVisible())

    # Check restore menubar visibility
    app.commands.execute_command(action_id)
    assert viewer.window._qt_window.menuBar().isVisible()
    assert not viewer.window._qt_window._toggle_menubar_visibility
