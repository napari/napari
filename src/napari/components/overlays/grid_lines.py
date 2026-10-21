from pydantic import Field

from napari.components.overlays.base import SceneOverlay
from napari.layers.base._base_constants import Blending
from napari.utils.color import ColorValue


class GridLinesOverlay(SceneOverlay):
    """Grid lines overlay.

    Attributes
    ----------
    color : ColorValue or None
        Color of the grid lines, or None for automatic coloring.
    axis_labels : bool
        Whether to display axis labels.
    font_size : float, optional
        The font size (in points) of the text. Axis labels will be 50% bigger.
    tick_labels : bool
        Whether to display tick labels.
    n_ticks : int
        How many ticks labels should be displayed. This number will be
        targeted approximately, to prioritize nice round numbers.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    color: ColorValue | None = None
    axis_labels: bool = True
    tick_labels: bool = True
    n_ticks: int = Field(default=5, ge=2)
    font_size: float | None = None
    blending: Blending = Blending.TRANSLUCENT_NO_DEPTH
