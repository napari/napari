"""Scale bar model."""

import warnings

from pydantic import Field, PrivateAttr

from napari.components.overlays.base import TiledCanvasOverlay
from napari.utils.color import ColorValue


class ScaleBarOverlay(TiledCanvasOverlay):
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
    font_size : float
        The font size (in points) of the text.
    length : Optional[float]
        Fixed length of the scale bar in physical units. If set to `None`,
        it is determined automatically based on zoom level.
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
    def unit(self, value: str | None) -> None:
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
