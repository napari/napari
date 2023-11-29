from qtpy.QtCore import QRect

from napari.settings import get_settings


class ScreenMock:
    def __init__(self):
        self._geometry = QRect(0, 0, 1000, 1000)

    def geometry(self):
        return self._geometry


def screen_at(point):
    if point.x() < 0 or point.y() < 0 or point.x() > 1000 or point.y() > 1000:
        return None
    return ScreenMock()


def test_singlescreen_window_settings(make_napari_viewer, monkeypatch):
    """Test whether valid screen position is returned even after disconnected secondary screen."""

    monkeypatch.setattr(
        "napari._qt.qt_main_window.QApplication.screenAt", screen_at
    )
    settings = get_settings()
    viewer = make_napari_viewer()
    default_window_position = (
        viewer.window._qt_window.x(),
        viewer.window._qt_window.y(),
    )

    # Valid position
    settings.application.window_position = (60, 50)
    window_position = viewer.window._qt_window._load_window_settings()[2]
    assert window_position == (60, 50)

    # Invalid left of screen
    settings.application.window_position = (0, -400)
    window_position = viewer.window._qt_window._load_window_settings()[2]
    assert window_position == default_window_position

    # Invalid right of screen
    settings.application.window_position = (0, 40000)
    window_position = viewer.window._qt_window._load_window_settings()[2]
    assert window_position == default_window_position
