from napari.components.overlays.axes import SceneAxesOverlay


def test_axes():
    """Test creating axes object"""
    axes = SceneAxesOverlay()
    assert axes is not None
