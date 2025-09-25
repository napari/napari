from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.text import Text
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

        self.overlay.events.text.connect(self._on_text_change)
        self.overlay.events.color.connect(self._on_color_change)
        self.overlay.events.font_size.connect(self._on_position_change)

        self.reset()

    def _on_text_change(self):
        self.node.text = self.overlay.text
        self._on_position_change()

    def _on_visible_change(self):
        # ensure that dpi is updated when the scale bar is visible
        # this does not need to run _on_position_change because visibility
        # is already connected to the canvas callback by the canvas itself
        self._on_text_change()
        return super()._on_visible_change()

    def _on_color_change(self):
        self.node.color = self.overlay.color

    def _on_position_change(self, event=None):
        position = self.overlay.position
        anchors = ('left', 'bottom')
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

        self.x_size, self.y_size = self.node.get_width_height()

        # depending on the canvas position, we need to change the position of the anchor itself
        # to ensure the text aligns properly e.g. left when on the left, and right when on the right
        x = y = 0.0
        if anchors[0] == 'right':
            x = self.x_size
        elif anchors[0] == 'center':
            x = self.x_size / 2

        if anchors[1] == 'top':
            y = self.y_size

        self.node.pos = (x, y)

        super()._on_position_change()

    def reset(self):
        super().reset()
        self._on_text_change()
        self._on_color_change()
