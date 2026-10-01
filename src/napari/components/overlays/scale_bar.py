"""Scale bar model."""

from pydantic import Field

from napari.components.overlays.base import CanvasOverlay
from napari.utils.color import ColorValue
from napari.utils.migrations import deprecation_warning


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
    length: float | None = None

    @property
    def unit(self) -> None:
        deprecation_warning(
            name='ScaleBar.unit',
            replacement='`Layer.units`',
            since='0.8.0',
            window='2026-Q4',
            details=(
                'It now always returns None, because units are computed from '
                'the layers in the layerlist.'
            ),
            stacklevel=4,
        )
        return None

    @unit.setter
    def unit(self, value: str | None) -> None:
        deprecation_warning(
            name='Setting `unit` on the ScaleBar model',
            replacement='`Layer.units`',
            since='0.8.0',
            window='2026-Q4',
            details=(
                'It no longer has any effect, because units are computed from '
                'the layers in the layerlist.'
            ),
            stacklevel=4,
        )
