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

        self._last_mouse_pos = None

        self.overlay.events.size.connect(self._on_size_change)
        self.node.events.canvas_change.connect(self._on_canvas_change)
        self.viewer.events.mouse_over_canvas.connect(
            self._on_mouse_over_canvas
        )
        # no need to connect position, since that's in the base classes of CanvasOverlay

        self.reset()

    def _on_position_change(self, event=None):
        self._set_position(self.overlay.position)

    def _on_size_change(self, event=None):
        self._white_circle.radius = self.overlay.size / 2
        self._black_circle.radius = self._white_circle.radius - 1

    def _on_visible_change(self):
        if self._last_mouse_pos is not None:
            self._set_position(self._last_mouse_pos)
        self.node.visible = (
            self.overlay.visible and self.viewer.mouse_over_canvas
        )

    def _on_mouse_move(self, event):
        self._last_mouse_pos = event.pos
        if self.overlay.visible:
            self.overlay.position = event.pos.tolist()

    def _set_position(self, pos):
        self.node.transform.translate = [pos[0], pos[1], 0, 0]

    def _on_canvas_change(self, event):
        if event.new is not None:
            event.new.events.mouse_move.connect(self._on_mouse_move)
        if event.old is not None:
            event.old.events.mouse_move.disconnect(self._on_mouse_move)

    def _on_mouse_over_canvas(self):
        self.node.visible = (
            self.overlay.visible and self.viewer.mouse_over_canvas
        )

    def reset(self):
        super().reset()
        self._on_size_change()
