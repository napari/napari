from napari.components.overlays.base import SceneOverlay


class LabelsPolygonOverlay(SceneOverlay):
    """Overlay that displays a polygon on a scene.

    Attributes
    ----------
    points : list
        A list of (x, y) coordinates of the vertices of the polygon.
    color : tuple
        A tuple representing the RGBA color of the polygon.
        Opacity only applies to the fill color of the polygon.
        Borders have the same color, but they are always opaque.
    dims_order : tuple
        A tuple representing the order of the dimensions in the scene.
    """

    points: list = []
    color: tuple = (1, 1, 1, 0.3)
    dims_order: tuple = (0, 1)
