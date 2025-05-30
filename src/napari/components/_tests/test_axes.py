from napari.components.overlays.axes import AxesOverlay


def test_axes():
    """Test creating axes object"""
    axes = AxesOverlay()
    assert axes is not None
