from vispy.scene.visuals import Text

from ...components._viewer_constants import CanvasPosition
from .base import VispyCanvasOverlay


class VispyTextOverlay(VispyCanvasOverlay):
    """Text overlay."""

    def __init__(self, **kwargs):
        super().__init__(node=Text(pos=(0, 0)), **kwargs)

        self.node.font_size = self.overlay.font_size
        self.node.anchors = ("left", "top")

        self.overlay.events.text.connect(self._on_text_change)
        self.overlay.events.color.connect(self._on_color_change)
        self.overlay.events.font_size.connect(self._on_color_change)

        self._on_visible_change()
        self._on_text_change()
        self._on_color_change()
        self._on_font_size_change()
        self._on_position_change()

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
