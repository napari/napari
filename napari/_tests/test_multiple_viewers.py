def test_multi_viewers_dont_clash(make_napari_viewer, qtbot):
    v1 = make_napari_viewer(title='v1')
    v2 = make_napari_viewer(title='v2')
    assert not v1.grid.enabled
    assert not v2.grid.enabled

    v1.window.activate()  # a click would do this in the actual gui
    v1.window._qt_viewer.viewerButtons.gridViewButton.click()

    assert not v2.grid.enabled
    assert v1.grid.enabled
