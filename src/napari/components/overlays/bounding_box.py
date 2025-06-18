from napari._pydantic_compat import Field
from napari.components.overlays.base import SceneOverlay
from napari.utils.color import ColorValue


class BoundingBoxOverlay(SceneOverlay):
    """
    Bounding box overlay to indicate layer boundaries.

    Attributes
    ----------
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
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    lines: bool = True
    line_thickness: float = 1
    line_color: ColorValue = Field(default_factory=lambda: ColorValue('red'))
    points: bool = True
    point_size: float = 5
    point_color: ColorValue = Field(default_factory=lambda: ColorValue('blue'))
