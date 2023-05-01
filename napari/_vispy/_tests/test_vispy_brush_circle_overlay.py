from napari._vispy.overlays.brush_circle import VispyBrushCircleOverlay
from napari.components import ViewerModel
from napari.components.overlays import BrushCircleOverlay


def test_vispy_brush_circle_overlay():
    brush_circle_model = BrushCircleOverlay()
    viewer = ViewerModel()

    vispy_brush_circle = VispyBrushCircleOverlay(
        viewer=viewer, overlay=brush_circle_model
    )
    brush_circle_model.size = 100
    brush_circle_model.position = 10, 20

    assert vispy_brush_circle._white_circle.radius == 50
    assert vispy_brush_circle._black_circle.radius == 49
