from napari.components.scale_bar import ScaleBar


def test_scale_bar():
    """Test creating scale bar object"""
    scale_bar = ScaleBar()
    assert scale_bar is not None
