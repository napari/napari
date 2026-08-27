from pydantic import Field

from napari.components.overlays.base import SceneOverlay
from napari.layers import Labels


class LabelsPolygonOverlay(SceneOverlay):
    """Overlay that displays a polygon on a scene.

    This overlay was created for drawing polygons on Labels layers. It handles
    the following mouse events to update the overlay:
    - Mouse move: Continuously redraw the latest polygon point with the current
    mouse position.
    - Mouse press (left button): Adds the current mouse position as a new
    polygon point.
    - Mouse double click (left button): If there are at least three points in
    the polygon and the double-click position is within completion_radius
    from the first vertex, the polygon will be painted in the image using the
    current label.
    - Mouse press (right button): Removes the most recent polygon point from
    the list.

    Attributes
    ----------
    enabled : bool
        Controls whether the overlay is activated.
    points : list
        A list of (x, y) coordinates of the vertices of the polygon.
    use_double_click_completion_radius : bool
        Whether double-click to complete drawing the polygon requires being within
        completion_radius of the first point.
    completion_radius : int | float
        Defines the radius from the first polygon vertex within which
        the drawing process can be completed by a left double-click.
    position : CanvasPosition
        The position of the overlay in the canvas.
    box : bool
        Whether the background box is visible or not.
    box_color : ColorValue or None
        Background box color. If unset, it defaults to the canvas color.
    gridded : bool
        The overlay will be duplicated across all grid cells in gridded mode.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    blending : Blending
        One of a list of preset blending modes that determines how RGB and
        alpha values of the overlay get mixed with the visuals below.
    """

    enabled: bool = False
    points: list = Field(default_factory=list)
    use_double_click_completion_radius: bool = False
    completion_radius: int = 20

    def add_polygon_to_labels(self, layer: Labels) -> None:
        if len(self.points) > 2:
            layer.paint_polygon(self.points, layer.selected_label)
        self.points = []
