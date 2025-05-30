from napari._pydantic_compat import Field
from napari.components.overlays.base import SceneOverlay
from napari.layers.base._base_constants import Blending
from napari.utils.color import ColorValue


class GridLinesOverlay(SceneOverlay):
    """Grid lines overlay.

    Attributes
    ----------
    color : ColorValue
        Color of the grid lines, or None for automatic coloring.
    labels : bool
        Whether to display ticks and tick labels.
    n_labels : int
        How many ticks and labels should be displayed. This number will be
        targeted approximately, to prioritize nice round numbers.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    color: ColorValue | None = None
    labels: bool = True
    n_labels: int = Field(5, ge=2)
    blending: Blending = Blending.TRANSLUCENT_NO_DEPTH
