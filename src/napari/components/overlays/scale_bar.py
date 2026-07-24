"""Scale bar model."""

import warnings

from pydantic import Field

from napari.components.overlays.base import CanvasOverlay
from napari.utils.color import ColorValue


class ScaleBarOverlay(CanvasOverlay):
    """Scale bar indicating size in world coordinates.

    Attributes
    ----------
    colored : bool
        If scale bar are colored or not. If colored then
        default color is magenta. If not colored than
        scale bar color is the opposite of the canvas
        background or the background box.
    color : ColorValue
        Scalebar and text color.
        See ``ColorValue.validate`` for supported values.
    ticks : bool
        If scale bar has ticks at ends or not.
    background_color : np.ndarray
        Background color of canvas. If scale bar is not colored
        then it has the color opposite of this color.
    font_size : float
        The font size (in points) of the text.
    box : bool
        If background box is visible or not.
    box_color : Optional[str | array-like]
        Background box color.
        See ``ColorValue.validate`` for supported values.
    length : Optional[float]
        Fixed length of the scale bar in physical units. If set to `None`,
        it is determined automatically based on zoom level.
    position : CanvasPosition
        The position of the overlay in the canvas.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    colored: bool = False
    color: ColorValue = Field(default_factory=lambda: ColorValue([1, 0, 1, 1]))
    ticks: bool = True
    font_size: float = 10
    length: float | None = None

    @property
    def unit(self) -> None:
        warnings.warn(
            'ScaleBar.unit is deprecated and now always returns None. '
            'This attribute will be removed in 0.9.0.\n'
            'Units are instead computed from the layers in the layerlist. '
            'Use `Layer.units` to set units for each layer.',
            category=FutureWarning,
            stacklevel=4,
        )
        return None

    @unit.setter
    def unit(self, value: str | None) -> None:
        warnings.warn(
            'Setting unit on the ScaleBar model is deprecated and no longer has any effect. '
            'This attribute will be removed in 0.9.0.\n'
            'Units are instead computed from the layers in the layerlist. '
            'Use `Layer.units` to set units for each layer.',
            category=FutureWarning,
            stacklevel=4,
        )
