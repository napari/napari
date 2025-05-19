from napari.components.overlays.scale_bar import ScaleBarOverlay


def test_scale_bar():
    """Test creating scale bar object"""
    scale_bar = ScaleBarOverlay()
    assert scale_bar is not None


def test_scale_bar_fixed_length():
    """Test creating scale bar object with fixed length"""
    scale_bar = ScaleBarOverlay(length=50)
    assert scale_bar.length == 50
