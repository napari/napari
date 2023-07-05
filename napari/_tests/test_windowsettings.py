from napari.settings import get_settings


def test_singlescreen_window_settings(make_napari_viewer):
    """Test whether valid screen position is returned even after disconnected secondary screen."""
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
