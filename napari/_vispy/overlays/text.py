from vispy.scene.visuals import Text

from ...components._viewer_constants import Position
from ...utils.translations import trans
from .base import VispyBaseOverlay


class VispyTextOverlay(VispyBaseOverlay):
    """Text overlay."""

    def __init__(self, **kwargs):
        node = Text(pos=(0, 0), anchor_x='center', anchor_y='center')

        super().__init__(node=node, **kwargs)

        self.overlay.events.text.connect(self._on_text_change)
        self.overlay.events.color.connect(self._on_color_change)
        self.overlay.events.font_size.connect(self._on_font_size_change)
        self.overlay.events.position.connect(self._on_position_change)
        self.node.parent.events.resize.connect(self._on_position_change)

        self.reset()

    def _on_text_change(self):
        """Change text value."""
        self.node.text = self.overlay.text
        self.node.update()

    def _on_color_change(self):
        """Update text size and color."""
        self.node.color = self.overlay.color
        self.node.update()

    def _on_font_size_change(self):
        self.node.font_size = self.overlay.font_size
        self.node.update()

    def _on_position_change(self, event=None):
        """Change position of text visual."""
        position = self.overlay.position
        x_offset, y_offset = 10, 5
        canvas_size = list(self.node.canvas.size)

        if position == Position.TOP_LEFT:
            transform = [x_offset, y_offset, 0, 0]
            anchors = ("left", "bottom")
        elif position == Position.TOP_RIGHT:
            transform = [canvas_size[0] - x_offset, y_offset, 0, 0]
            anchors = ("right", "bottom")
        elif position == Position.TOP_CENTER:
            transform = [canvas_size[0] // 2, y_offset, 0, 0]
            anchors = ("center", "bottom")
        elif position == Position.BOTTOM_RIGHT:
            transform = [
                canvas_size[0] - x_offset,
                canvas_size[1] - y_offset,
                0,
                0,
            ]
            anchors = ("right", "top")
        elif position == Position.BOTTOM_LEFT:
            transform = [x_offset, canvas_size[1] - y_offset, 0, 0]
            anchors = ("left", "top")
        elif position == Position.BOTTOM_CENTER:
            transform = [canvas_size[0] // 2, canvas_size[1] - y_offset, 0, 0]
            anchors = ("center", "top")
        else:
            raise ValueError(
                trans._(
                    "Position {position} is not recognized.", position=position
                )
            )

        self.node.transform.translate = transform
        if self.node.anchors != anchors:
            self.node.anchors = anchors

        self.node.update()

    def reset(self):
        super().reset()
        self._on_text_change()
        self._on_color_change()
        self._on_font_size_change()
        self._on_position_change()
