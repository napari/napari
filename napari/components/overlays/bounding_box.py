from napari.components.overlays.base import SceneOverlay
from napari.utils.color import ColorValue


class BoundingBoxOverlay(SceneOverlay):
    """
    Bounding box overlay to indicate layer boundaries.

    Attributes
    ----------
    visible : bool
        If the bounding box is visible or not.
    lines : bool
        Whether to show the lines of the bounding box.
    line_thickness : float
        Thickness of the lines in canvas pixels.
    line_color : ColorValue
        Color of the lines.
    points : bool
        Whether to show the vertices of the bounding box as points.
    point_size : float
        Size of the points in canvas pixels.
    point_color : ColorValue
        Color of the points.
    """

    lines: bool = True
    line_thickness: float = 1
    line_color: ColorValue = 'red'
    points: bool = True
    point_size: float = 5
    point_color: ColorValue = 'blue'
