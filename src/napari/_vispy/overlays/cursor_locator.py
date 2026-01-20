import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.cursor_locator import CursorLocator


class VispyCursorLocatorOverlay(ViewerOverlayMixin, VispySceneOverlay):
    """Overlay indicating the position of the cursor in the world."""

    def __init__(self, *, viewer, overlay, parent=None) -> None:
        super().__init__(
            node=CursorLocator(), viewer=viewer, overlay=overlay, parent=parent
        )
        self.viewer.cursor.events.position.connect(self._on_cursor_move)

        self.reset()

    def _on_cursor_move(self):
        displayed = list(self.viewer.dims.displayed[::-1])
        if len(displayed) == 2:
            displayed = np.concat([displayed, [0]])
        self.node.set_position(
            np.array(self.viewer.cursor.position)[displayed]
        )

    def reset(self):
        super().reset()
        self._on_cursor_move()
