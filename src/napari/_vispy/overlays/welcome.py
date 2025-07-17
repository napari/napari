import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.welcome import Welcome
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.theme import get_theme


class VispyWelcomeOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    def __init__(self, *, viewer, overlay, parent=None) -> None:
        super().__init__(
            node=Welcome(), viewer=viewer, overlay=overlay, parent=parent
        )
        self.viewer.events.theme.connect(self._on_theme_change)
        self.viewer.layers.events.connect(self._on_visible_change)

        self.node.canvas.native.resized.connect(self._on_position_change)

        self.reset()

    def _on_position_change(self, event=None):
        if self.node.canvas is not None:
            x, y = np.array(self.node.canvas.size) / 2
            self.node.transform.translate = (x, y, 0, 0)

    def _on_theme_change(self):
        if self.node.parent is not None and self.node.parent.canvas.bgcolor:
            background_color = self.node.parent.canvas.bgcolor.rgba
        else:
            background_color = get_theme(self.viewer.theme).canvas.as_hex()
            background_color = transform_color(background_color)[0]
        color = np.subtract(1, background_color)
        color[-1] = background_color[-1]
        self.node.set_color(color)

    def _on_visible_change(self):
        self.node.visible = self.overlay.visible and not self.viewer.layers

    def _on_text_change(self):
        self.node.set_text()

    def reset(self):
        super().reset()
        self._on_theme_change()
        self._on_text_change()
