from psygnal import EventedModel
from pydantic import ConfigDict

from napari.components._viewer_constants import CanvasPosition
from napari.layers.base._base_constants import Blending
from napari.utils.color import ColorValue


class Overlay(EventedModel):
    """
    Overlay evented model.

    An overlay is a renderable entity meant to display additional information
    on top of the layer data, but is not data per se.
    For example: a scale bar, a color bar, axes, bounding boxes, etc.

    Attributes
    ----------
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

    model_config = EventedModel.model_config | ConfigDict(  # type: ignore[typeddict-unknown-key]
        validate_assignment=True,
        allow_property_setters=True,
        guess_property_dependencies=True,
    )

    visible: bool = False
    opacity: float = 1
    order: int = 10**6
    blending: Blending

    def __hash__(self):
        return id(self)


class CanvasOverlay(Overlay):
    """
    Canvas overlay model.

    Canvas overlays live in canvas space; they do not live in the 2- or 3-dimensional
    scene being rendered, but in the 2D space of the screen.

    Attributes
    ----------
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

    blending: Blending = Blending.TRANSLUCENT_NO_DEPTH


class TiledCanvasOverlay(CanvasOverlay):
    """
    Canvas overlay model.

    Canvas overlays live in canvas space; they do not live in the 2- or 3-dimensional
    scene being rendered, but in the 2D space of the screen.
    Tiled canvas overlays are not rendered freely on the canvas: they are tiled
    around the edges.

    Attributes
    ----------
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

    position: CanvasPosition = CanvasPosition.BOTTOM_RIGHT
    blending: Blending = Blending.TRANSLUCENT_NO_DEPTH
    box: bool = True
    box_color: ColorValue | None = None
    gridded: bool = False


class SceneOverlay(Overlay):
    """
    Scene overlay model.

    Scene overlays live in the 2- or 3-dimensional space of the rendered data.
    For example: bounding boxes, data grids, etc.

    Attributes
    ----------
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

    blending: Blending = Blending.TRANSLUCENT
