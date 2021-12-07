from napari import Viewer


def test_multi_viewers_dont_clash(qapp):
    v1 = Viewer(show=False, title='v1')
    v2 = Viewer(show=False, title='v2')
    assert not v1.grid.enabled
    assert not v2.grid.enabled

    v1.window.activate()  # a click would do this in the actual gui
    v1.window._qt_viewer.viewerButtons.gridViewButton.click()

    assert not v2.grid.enabled
    assert v1.grid.enabled

    v1.close()
    v2.close()
