from napari.components.overlays import SceneOverlay
from napari.utils.color import ColorValue


class CursorOverlay(SceneOverlay):
    """Used as the key for overlay_to_visual's overlay dict"""

    color: ColorValue = (0, 1, 0, 1)
    size: int = 10
