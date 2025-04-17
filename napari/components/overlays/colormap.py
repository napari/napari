from napari._pydantic_compat import Field
from napari.components.overlays.base import CanvasOverlay
from napari.utils.color import ColorValue


class ColormapOverlay(CanvasOverlay):
    """Colormap legend overlay.

    Attributes
    ----------
    position : CanvasPosition
        The position of the overlay in the canvas.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    size: tuple[float, float] = 50, 250
    ticks: bool = True
    n_ticks: int = 4
    tick_length: float = 5
    font_size: float = 7
    color: ColorValue = Field(default_factory=lambda: ColorValue('white'))
