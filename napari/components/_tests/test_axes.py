from napari.components.axes import Axes


def test_axes():
    """Test creating axes object"""
    axes = Axes()
    assert axes is not None
