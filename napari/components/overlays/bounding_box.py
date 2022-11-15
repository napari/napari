from napari.components.overlays.base import SceneOverlay
from napari.utils.color import ColorValue


class BoundingBoxOverlay(SceneOverlay):
    lines: bool = True
    line_thickness = 1
    line_color: ColorValue = 'red'
    points: bool = True
    point_size = 5
    point_color: ColorValue = 'blue'
