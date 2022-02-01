from unittest.mock import patch

from napari._qt.qt_main_window import Window, _QtMainWindow
from napari.utils.theme import (
    _themes,
    get_theme,
    register_theme,
    unregister_theme,
)


def test_current_viewer(make_napari_viewer, qapp):
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


@patch.object(Window, "_theme_icon_changed")
@patch.object(Window, "_remove_theme")
@patch.object(Window, "_add_theme")
def test_update_theme(
    mock_add_theme,
    mock_remove_theme,
    mock_icon_changed,
    make_napari_viewer,
    qapp,
):
    viewer = make_napari_viewer()

    blue = get_theme("dark", False)
    blue.name = "blue"
    register_theme("blue", blue)

    # triggered when theme was added
    mock_add_theme.assert_called()
    mock_remove_theme.assert_not_called()

    unregister_theme("blue")
    # triggered when theme was removed
    mock_remove_theme.assert_called()

    mock_icon_changed.assert_not_called()
    viewer.theme = "light"
    theme = _themes["light"]
    theme.icon = "#FF0000"
    mock_icon_changed.assert_called()


def test_lazy_console(make_napari_viewer):
    v = make_napari_viewer()
    assert v.window._qt_viewer._console is None
    v.update_console({"test": "test"})
    assert v.window._qt_viewer._console is None
