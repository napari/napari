from napari import Viewer
from napari.settings import get_settings


def test_singlescreen_window_settings(qtbot):
    settings = get_settings()
    viewer = Viewer(show=False)

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
    assert window_position == (0, 0)

    # Invalid right of screen
    settings.application.window_position = (0, 40000)
    (
        _,
        _,
        window_position,
        _,
        _,
    ) = viewer.window._qt_window._load_window_settings()
    assert window_position == (0, 0)
