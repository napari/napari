"""Scale bar model."""

import warnings

from pydantic import Field, PrivateAttr

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
    _unit: str | None = PrivateAttr(default=None)
    length: float | None = None

    @property
    def unit(self) -> str | None:
        return self._unit

    @unit.setter
    def unit(self, value: str | None):
        if value is not None:
            warnings.warn(
                'Setting unit on the ScaleBar model is deprecated. Units will instead be computed from '
                'the layers in the layerlist. To silence this warning, leave scale_bar unit as `None`, '
                'and use `Layer.units` to set units for each layer. Starting in v0.8.0, setting '
                'ScaleBar.unit will no longer have an effect. Starting from v0.9.0, it will be '
                'removed and raise an exception.',
                category=FutureWarning,
                stacklevel=4,
            )
        self._unit = value
        self.events.unit(self._unit)
