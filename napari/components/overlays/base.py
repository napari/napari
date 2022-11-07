from napari.components._viewer_constants import CanvasPosition
from napari.utils.events import EventedModel


class BaseOverlay(EventedModel):
    """
    Base class for overlay evented models.

    An overlay is a renderable entity meant to display additional information
    on top of the layer data, but is not data per se.
    For example: a scale bar, a color bar, axes, bounding boxes, etc.
    """

    visible: bool = False
    opacity: float = 1
    order: int = 1e6


class CanvasOverlay(BaseOverlay):
    """
    Canvas overlay model.

    Canvas overlays live in canvas space; they do not live in the 2- or 3-dimensional
    scene being rendered, but in the 2D space of the screen.
    For example: scale bars, colormap bars, etc.
    """

    position: CanvasPosition = CanvasPosition.BOTTOM_RIGHT


class SceneOverlay(BaseOverlay):
    """
    Scene overlay model.

    Scene overlays live in the 2- or 3-dimensional space of the rendered data.
    For example: bounding boxes, data grids, etc.
    """

    # TODO: should transform live here?
