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

        self.node.anchors = ('left', 'bottom')
        self._vertices_size = (0, 0)

        self.overlay.events.text.connect(self._on_text_change)
        self.overlay.events.color.connect(self._on_color_change)
        self.overlay.events.font_size.connect(self._on_position_change)

        self.reset()

    def _on_text_change(self):
        self.node.text = self.overlay.text
        self._on_position_change()

    def _on_color_change(self):
        self.node.color = self.overlay.color

    def _on_position_change(self, event=None):
        super()._on_position_change()
        position = self.overlay.position
        if position == CanvasPosition.TOP_LEFT:
            anchors = ('left', 'bottom')
        elif position == CanvasPosition.TOP_RIGHT:
            anchors = ('right', 'bottom')
        elif position == CanvasPosition.TOP_CENTER:
            anchors = ('center', 'bottom')
        elif position == CanvasPosition.BOTTOM_RIGHT:
            anchors = ('right', 'top')
        elif position == CanvasPosition.BOTTOM_LEFT:
            anchors = ('left', 'top')
        elif position == CanvasPosition.BOTTOM_CENTER:
            anchors = ('center', 'top')

        self.node.anchors = anchors
        self.node.font_size = self.overlay.font_size

        vert_buffer = self.node._vertices_data
        if vert_buffer is not None:
            pos = vert_buffer['a_position']
            tl = pos.min(axis=0)
            br = pos.max(axis=0)
            self._vertices_size = (br[0] - tl[0]), (br[1] - tl[1])

        self.x_size = (
            self._vertices_size[0] * self.overlay.font_size * 1.3
        )  # magic?
        self.y_size = self._vertices_size[1] * self.overlay.font_size

        x = y = 0
        if 'right' in anchors:
            x = self.x_size
        elif 'center' in anchors:
            x = self.x_size / 2
        if 'top' in anchors:
            y = self.y_size / 2

        self.node.pos = (x, y)

    def reset(self):
        self._on_text_change()
        super().reset()
        self._on_color_change()
