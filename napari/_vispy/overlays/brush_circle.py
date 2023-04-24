from vispy.scene.visuals import Compound, Ellipse

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay


class VispyBrushCircleOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    def __init__(self, *, viewer, overlay, parent=None):
        self._white_circle = Ellipse(
            center=(0, 0),
            color=(0, 0, 0, 0.0),
            border_color='white',
            border_method='agg',
        )
        self._black_circle = Ellipse(
            center=(0, 0),
            color=(0, 0, 0, 0.0),
            border_color='black',
            border_method='agg',
        )

        super().__init__(
            node=Compound([self._white_circle, self._black_circle]),
            viewer=viewer,
            overlay=overlay,
            parent=parent,
        )

        self.overlay.events.size.connect(self._on_size_change)
        # no need to connect position, since that's in the base classes of CanvasOverlay

        self.reset()

    def _on_position_change(self, event=None):
        if self.node.canvas is None:
            return
        x, y = self.overlay.position
        self.node.transform.translate = [x, y, 0, 0]

    def _on_size_change(self, event=None):
        self._white_circle.radius = self.overlay.size / 2
        self._black_circle.radius = self._white_circle.radius - 1

    def reset(self):
        super().reset()
        self._on_size_change()
