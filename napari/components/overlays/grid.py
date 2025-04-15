from napari._pydantic_compat import Field
from napari.components.overlays.base import SceneOverlay
from napari.utils.color import ColorValue


class GridOverlay(SceneOverlay):
    """Grid lines overlay.

    Attributes
    ----------
    color : ColorValue
        Color of the grid lines.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    color: ColorValue = Field(default_factory=lambda: ColorValue('white'))
