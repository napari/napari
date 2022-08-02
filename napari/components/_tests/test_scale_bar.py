from napari.components.overlays.scale_bar import ScaleBarOverlay


def test_scale_bar():
    """Test creating scale bar object"""
    scale_bar = ScaleBarOverlay()
    assert scale_bar is not None
