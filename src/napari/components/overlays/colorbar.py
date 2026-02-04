from napari.components._viewer_constants import CanvasPosition
from napari.components.overlays.base import CanvasOverlay
from napari.utils.color import ColorValue


class ColorBarOverlay(CanvasOverlay):
    """ColorBar legend overlay.

    Attributes
    ----------
    position : CanvasPosition
        The position of the overlay in the canvas.
    visible : bool
        If the overlay is visible or not. Inherited from Overlay.
    opacity : float
        The opacity of the overlay. 0 is fully transparent. Inherited from Overlay.
    order : int
        The rendering order of the overlay: lower numbers get rendered first. Inherited from Overlay.
    color : ColorValue | None
        The color of the outline and ticks of the colorbar.
    size : tuple[float, float]
        The size of the colorbar in pixels (width, height).
    tick_length : float
        The length of the ticks in pixels.
    font_size : float
        The font size of the tick labels.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the colorbar visual get mixed with other visuals. Defaults to 'translucent_no_depth'. Inherited from CanvasOverlay.
    gridded : bool
        The overlay will be duplicated across all grid cells in gridded mode. Inherited from CanvasOverlay.
    """

    color: ColorValue | None = None
    size: tuple[float, float] = 25, 150
    tick_length: float = 5
    font_size: float = 10
    position: CanvasPosition = CanvasPosition.TOP_RIGHT
