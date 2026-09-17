from pydantic import Field

from napari.components._viewer_constants import CanvasPosition
from napari.components.overlays.base import TiledCanvasOverlay
from napari.utils.color import ColorValue


class ColorBarOverlay(TiledCanvasOverlay):
    """ColorBar legend overlay.

    Attributes
    ----------
    color : ColorValue | None
        The color of the outline and ticks of the colorbar.
    size : tuple[float, float]
        The size of the colorbar in pixels (width, height).
    tick_length : float
        The length of the ticks in pixels.
    font_size : float
        The font size of the tick labels.
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

    color: ColorValue | None = None
    size: tuple[float, float] = 25, 150
    tick_length: float = 5
    font_size: float = 10
    position: CanvasPosition = CanvasPosition.TOP_RIGHT
    colormanager_attribute: str | None = Field(
        default=None, frozen=True, repr=False
    )
