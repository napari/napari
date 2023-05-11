from napari.components.overlays.base import SceneOverlay
from napari.layers import Labels


class LabelsPolygonOverlay(SceneOverlay):
    """Overlay that displays a polygon on a scene.

        It handles the following mouse events to update the overlay:
        - Mouse move: Continuously redraw the latest polygon point with the current mouse position.
        - Mouse press (left button): Adds the current mouse position as a new polygon point.
        - Mouse double click (left button): If there are at least three points in the polygon,
                      the polygon is painted in the image using the current label.
        - Mouse press (right button): Removes the most recent polygon point from the list.

    Attributes
    ----------
    enabled : bool
        Controls whether the overlay is activated.
    points : list
        A list of (x, y) coordinates of the vertices of the polygon.
    color : tuple
        A tuple representing the RGBA color of the polygon.
        Opacity only applies to the fill color of the polygon.
        Borders have the same color, but they are always opaque.
    """

    enabled: bool = False
    points: list = []
    color: tuple = (1, 1, 1, 0.3)

    def add_polygon_to_labels(self, layer: Labels) -> None:
        if len(self.points) > 2:
            layer.paint_polygon(self.points, layer.selected_label)
        self.points = []
