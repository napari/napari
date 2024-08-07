from napari.components.overlays.scale_bar import ScaleBarOverlay


def test_scale_bar():
    """Test creating scale bar object"""
    scale_bar = ScaleBarOverlay()
    assert scale_bar is not None


def test_scale_bar_fixed_width():
    """Test creating scale bar object with fixed width"""
    scale_bar = ScaleBarOverlay(fixed_width=50)
    assert scale_bar.fixed_width == 50
