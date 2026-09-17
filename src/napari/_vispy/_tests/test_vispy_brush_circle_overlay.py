import numpy as np

from napari._vispy.overlays.brush_circle import VispyBrushCircleOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.components.overlays import BrushCircleOverlay
from napari.layers import Labels


def test_vispy_brush_circle_overlay():
    brush_circle = BrushCircleOverlay()
    viewer = ViewerModel()
    labels = Labels(data=np.zeros((10, 10), dtype=np.int32))

    _ = VispyBrushCircleOverlay(
        layer=labels, viewer=viewer, overlay=brush_circle, font_info=FontInfo()
    )
    assert brush_circle.visible is False
