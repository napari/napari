from napari.components.overlays.brush_circle import BrushCircleOverlay


def test_brush_circle():
    """Test creating a brush circle overlay"""
    brush_circle = BrushCircleOverlay()
    assert brush_circle is not None
