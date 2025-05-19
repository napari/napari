from vispy.scene.visuals import Ellipse

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay


class VispyCursorOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    def __init__(self, *, viewer, overlay, parent=None):
        # the node argument for the base class is the vispy visual
        # note that the center is (0, 0), cause we handle the shift with transforms
        super().__init__(
            node=Ellipse(center=(0, 0)),
            viewer=viewer,
            overlay=overlay,
            parent=parent,
        )

        # we also need to connect events from the model to callbacks that update the visual
        self.overlay.events.color.connect(self._on_color_change)
        self.overlay.events.size.connect(self._on_size_change)
        self.overlay.events.position.connect(self._on_position_change)

        self.reset()

    def _on_color_change(self, event=None):
        self.node.color = self.overlay.color

    def _on_position_change(self, event=None):
        # we can overload the position changing to account for the size, so that the dot
        # always sticks to the edge; there are `offset` attributes specifically for this
        self.x_offset = self.y_offset = self.overlay.size / 2
        super()._on_position_change()

    def _on_size_change(self, event=None):
        self.node.radius = self.overlay.size / 2
        # trigger position update since the radius changed
        self._on_position_change()

    # we should always add all the new callbacks to the reset() method
    def reset(self):
        super().reset()
        self._on_color_change()
        self._on_size_change()
