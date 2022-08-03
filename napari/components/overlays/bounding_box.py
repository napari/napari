from ...utils.color import ColorValue
from .base import LayerOverlay


class BoundingBoxOverlay(LayerOverlay):
    lines: bool = True
    line_thickness = 1
    line_color: ColorValue = 'red'
    points: bool = True
    point_size = 5
    point_color: ColorValue = 'blue'
