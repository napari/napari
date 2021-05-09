"""Label visual."""
# Third-party imports
from vispy.scene.visuals import Text
from vispy.visuals.transforms import STTransform

# Local imports
from ..components._viewer_constants import TextOverlayPosition
from ..utils.translations import trans


class VispyTextVisual:
    """Text overlay visual."""

    def __init__(self, viewer, parent=None, order=1e6):
        self._viewer = viewer

        self.node = Text(pos=(0, 0), parent=parent)
        self.node.order = order
        self.node.transform = STTransform()
        self.node.font_size = self._viewer.text_overlay.font_size
        self.node.anchors = ("center", "center")

        self._viewer.text_overlay.events.visible.connect(
            self._on_visible_change
        )
        self._viewer.text_overlay.events.text.connect(self._on_data_change)
        self._viewer.text_overlay.events.color.connect(self._on_text_change)
        self._viewer.text_overlay.events.font_size.connect(
            self._on_text_change
        )
        self._viewer.text_overlay.events.position.connect(
            self._on_position_change
        )
        self._viewer.camera.events.zoom.connect(self._on_position_change)

        self._on_visible_change(None)
        self._on_data_change(None)
        self._on_text_change(None)
        self._on_position_change(None)

    def _on_visible_change(self, event):
        """Change text visibility."""
        self.node.visible = self._viewer.text_overlay.visible

    def _on_data_change(self, event):
        """Change text value."""
        self.node.text = self._viewer.text_overlay.text

    def _on_text_change(self, event):
        """Update text size and color."""
        self.node.font_size = self._viewer.text_overlay.font_size
        self.node.color = self._viewer.text_overlay.color

    def _on_position_change(self, event):
        """Change position of text visual."""
        position = self._viewer.text_overlay.position
        x_offset, y_offset = 10, 5
        canvas_size = list(self.node.canvas.size)

        if position == TextOverlayPosition.TOP_LEFT:
            transform = [x_offset, y_offset, 0, 0]
            anchors = ("left", "bottom")
        elif position == TextOverlayPosition.TOP_RIGHT:
            transform = [canvas_size[0] - x_offset, y_offset, 0, 0]
            anchors = ("right", "bottom")
        elif position == TextOverlayPosition.TOP_CENTER:
            transform = [canvas_size[0] // 2, y_offset, 0, 0]
            anchors = ("center", "bottom")
        elif position == TextOverlayPosition.BOTTOM_RIGHT:
            transform = [
                canvas_size[0] - x_offset,
                canvas_size[1] - y_offset,
                0,
                0,
            ]
            anchors = ("right", "top")
        elif position == TextOverlayPosition.BOTTOM_LEFT:
            transform = [x_offset, canvas_size[1] - y_offset, 0, 0]
            anchors = ("left", "top")
        elif position == TextOverlayPosition.BOTTOM_CENTER:
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
