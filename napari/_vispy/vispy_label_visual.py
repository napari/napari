"""Label visual"""
# Third-party imports
from vispy.scene.visuals import Text
from vispy.visuals.transforms import STTransform

# Local imports
from ..components._viewer_constants import Position


class VispyLabelVisual:
    """Label visual."""

    def __init__(self, viewer, parent=None, order=1e6):
        self._viewer = viewer

        self.node = Text(pos=(0, 0), parent=parent)
        self.node.order = order
        self.node.transform = STTransform()
        self.node.font_size = self._viewer.label.font_size
        self.node.anchors = ("center", "center")

        self._viewer.label.events.visible.connect(self._on_visible_change)
        self._viewer.label.events.text.connect(self._on_data_change)
        self._viewer.label.events.color.connect(self._on_data_change)
        self._viewer.label.events.font_size.connect(self._on_text_change)
        self._viewer.label.events.position.connect(self._on_position_change)
        self._viewer.camera.events.zoom.connect(self._on_position_change)

        self._on_visible_change(None)
        self._on_data_change(None)
        self._on_text_change(None)
        self._on_position_change(None)

    def _on_visible_change(self, _evt=None):
        """Change text visibility"""
        self.node.visible = self._viewer.label.visible

    def _on_data_change(self, _evt=None):
        """Change text value."""
        self.node.text = self._viewer.label.text

    def _on_text_change(self, _evt=None):
        """Update text size and color"""
        self.node.font_size = self._viewer.label.font_size
        self.node.color = self._viewer.label.color

    def _on_position_change(self, _evt=None):
        """Change position of text visual."""
        position = self._viewer.label.position
        x_offset, y_offset = 10, 5
        canvas_size = list(self.node.canvas.size)

        if position == Position.TOP_LEFT:
            transform = [x_offset, y_offset, 0, 0]
            anchors = ("left", "bottom")
        elif position == Position.TOP_RIGHT:
            transform = [canvas_size[0] - 5, y_offset, 0, 0]
            anchors = ("right", "bottom")
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
        else:
            raise ValueError(f"Position {position} is not recognized.")

        self.node.transform.translate = transform
        if self.node.anchors != anchors:
            self.node.anchors = anchors
