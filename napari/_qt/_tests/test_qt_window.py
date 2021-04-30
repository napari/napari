from napari import Viewer
from napari._qt.qt_main_window import _QtMainWindow


def test_active_viewer(make_napari_viewer, qapp):
    """Test that we can retrieve the "active" viewer instance easily.

    ... where "active" means it was the last viewer the user interacted with.
    """
    assert Viewer.active() is None
    assert _QtMainWindow.current() is None

    # when we create a new viewer it becomes accessible at Viewer.active()
    v1 = make_napari_viewer(title='v1')
    assert Viewer.active() == v1
    assert _QtMainWindow._instances == [v1.window._qt_window]
    assert _QtMainWindow.current() == v1.window._qt_window

    v2 = make_napari_viewer(title='v2')
    assert Viewer.active() == v2
    assert _QtMainWindow._instances == [
        v1.window._qt_window,
        v2.window._qt_window,
    ]

    # Viewer.active() will always give the most recently activated viewer.
    v1.window.activate()
    assert Viewer.active() == v1
    v2.window.activate()
    assert Viewer.active() == v2

    # The list remembers the z-order of previous viewers ...
    v2.close()
    assert Viewer.active() == v1
    assert _QtMainWindow._instances == [v1.window._qt_window]

    # and when none are left, Viewer.active() becomes None again
    v1.close()
    assert Viewer.active() is None
    assert _QtMainWindow._instances == []
    assert _QtMainWindow.current() is None
