from napari.components.overlays.base import SceneOverlay


class LabelsPolygonOverlay(SceneOverlay):
    points: list = []
    color: tuple = (1, 1, 1, 0.3)
    dims_order: tuple = (0, 1)
