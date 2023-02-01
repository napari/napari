from vispy.scene.visuals import Text

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari.components._viewer_constants import CanvasPosition


class VispyTextOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    """Text overlay."""

    def __init__(self, *, viewer, overlay, parent=None) -> None:
        super().__init__(
            node=Text(pos=(0, 0)),
            viewer=viewer,
            overlay=overlay,
            parent=parent,
        )

        self.node.font_size = self.overlay.font_size
        self.node.anchors = ("left", "top")

        self.overlay.events.text.connect(self._on_text_change)
        self.overlay.events.color.connect(self._on_color_change)
        self.overlay.events.font_size.connect(self._on_font_size_change)

        self.reset()

    def _on_text_change(self):
        self.node.text = self.overlay.text

    def _on_color_change(self):
        self.node.color = self.overlay.color

    def _on_font_size_change(self):
        self.node.font_size = self.overlay.font_size

    def _on_position_change(self, event=None):
        super()._on_position_change()
        position = self.overlay.position

        if position == CanvasPosition.TOP_LEFT:
            anchors = ("left", "bottom")
        elif position == CanvasPosition.TOP_RIGHT:
            anchors = ("right", "bottom")
        elif position == CanvasPosition.TOP_CENTER:
            anchors = ("center", "bottom")
        elif position == CanvasPosition.BOTTOM_RIGHT:
            anchors = ("right", "top")
        elif position == CanvasPosition.BOTTOM_LEFT:
            anchors = ("left", "top")
        elif position == CanvasPosition.BOTTOM_CENTER:
            anchors = ("center", "top")

        self.node.anchors = anchors

    def reset(self):
        super().reset()
        self._on_text_change()
        self._on_color_change()
        self._on_font_size_change()
