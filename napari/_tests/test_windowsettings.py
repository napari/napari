import gc
import platform

from napari import Viewer
from napari.settings import get_settings


def test_singlescreen_window_settings(qtbot):
    """Test whether valid screen position is returned even after disconnected secondary screen."""
    os = platform.system()
    settings = get_settings()
    viewer = Viewer(show=False)
    default_window_position = (
        (viewer.window._qt_window.x(), viewer.window._qt_window.y())
        if os == "Darwin"
        else (
            viewer.window._qt_window.y(),
            viewer.window._qt_window.x(),
        )
    )

    # Valid position
    settings.application.window_position = (50, 50)
    (
        _,
        _,
        window_position,
        _,
        _,
    ) = viewer.window._qt_window._load_window_settings()
    assert window_position == (50, 50)

    # Invalid left of screen
    settings.application.window_position = (0, -400)
    (
        _,
        _,
        window_position,
        _,
        _,
    ) = viewer.window._qt_window._load_window_settings()
    assert window_position == default_window_position

    # Invalid right of screen
    settings.application.window_position = (0, 40000)
    (
        _,
        _,
        window_position,
        _,
        _,
    ) = viewer.window._qt_window._load_window_settings()
    assert window_position == default_window_position
    viewer.close()
    qtbot.wait(50)
    gc.collect()
