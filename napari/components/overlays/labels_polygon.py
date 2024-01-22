from napari._pydantic_compat import Field
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
    double_click_completion : bool
        Whether drawing the polygon can be completed with a left mouse double-click.
    completion_radius : int
        Defines the radius from the first polygon vertex within which
        the drawing process can be completed by a left double-click.
    """

    enabled: bool = False
    points: list = Field(default_factory=list)
    double_click_completion: bool = True
    completion_radius: int = 20

    def add_polygon_to_labels(self, layer: Labels) -> None:
        if len(self.points) > 2:
            layer.paint_polygon(self.points, layer.selected_label)
        self.points = []
