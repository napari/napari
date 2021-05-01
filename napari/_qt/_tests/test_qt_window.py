from napari._qt.qt_main_window import _QtMainWindow


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
