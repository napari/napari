import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.cursor_locator import Crosshair


class VispyCursorLocatorOverlay(ViewerOverlayMixin, VispySceneOverlay):
    """Overlay indicating the position of the cursor in the world."""

    def __init__(self, *, viewer, overlay, parent=None, **kwargs) -> None:
        super().__init__(
            node=Crosshair(),
            viewer=viewer,
            overlay=overlay,
            parent=parent,
            **kwargs,
        )
        self.overlay.events.color.connect(self._on_color_change)
        self.overlay.events.gap.connect(self._on_gap_change)

        self.viewer.cursor.events.position.connect(self._on_position_change)

        self.reset()

    def _on_position_change(self):
        displayed = list(self.viewer.dims.displayed[::-1])
        if len(displayed) == 2:
            displayed = np.concat([displayed, [0]])
        self.node.position = np.array(self.viewer.cursor.position)[displayed]

    def _on_color_change(self):
        self.node.color = self.overlay.color

    def _on_gap_change(self):
        self.node.gap = self.overlay.gap

    def reset(self):
        super().reset()
        self._on_position_change()
        self._on_color_change()
        self._on_gap_change()
