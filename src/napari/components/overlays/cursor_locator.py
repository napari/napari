from napari.components.overlays.base import SceneOverlay
from napari.utils.color import ColorValue


class CursorLocatorOverlay(SceneOverlay):
    """
    Overlay that displays where the cursor is located in the world.
    """

    color: ColorValue = ColorValue('red')
    gap: float = 0.05
