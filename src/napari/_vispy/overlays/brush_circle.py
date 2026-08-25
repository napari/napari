from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vispy.scene.visuals import Compound, Ellipse

from napari._vispy.overlays.base import LayerOverlayMixin, VispyCanvasOverlay

if TYPE_CHECKING:
    from napari.components.overlays.brush_circle import BrushCircleOverlay
    from napari.layers.labels import Labels
    from napari.utils.events import Event


class VispyBrushCircleOverlay(LayerOverlayMixin, VispyCanvasOverlay):
    overlay: BrushCircleOverlay
    layer: Labels

    def __init__(self, **kwargs: Any) -> None:
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
            **kwargs,
        )

        self._last_mouse_pos = None

        self.layer.events.brush_size.connect(self._on_size_change)
        self.layer.events.brush_size_is_canvas.connect(self._on_size_change)
        self.viewer.scene.camera.events.zoom.connect(self._on_size_change)
        self.viewer.events.mouse_over_canvas.connect(
            self._on_mouse_over_canvas
        )
        # no need to connect position, since that's in the base classes of CanvasOverlay

        self.node.events.canvas_change.connect(self._on_canvas_change)
        self.reset()

        # manually connect this once and get the correct canvas
        if self.node.parent is not None:
            self.node.parent.scene.canvas.events.mouse_move.connect(
                self._on_mouse_move
            )

    def _on_position_change(self, event: Event | None = None) -> None:
        self._set_position(self.viewer.cursor.canvas_position)

    def _on_size_change(self, event: Event | None = None) -> None:
        if self.layer.brush_size_is_canvas:
            size = self.layer.brush_size
        else:
            size = self.layer._get_brush_size_canvas(
                self.viewer.scene.camera.zoom
            )
        self._white_circle.radius = size / 2
        self._black_circle.radius = self._white_circle.radius - 1

    def _on_visible_change(self) -> None:
        if self._last_mouse_pos is not None:
            self._set_position(self._last_mouse_pos)
        self.node.visible = (
            self.overlay.visible and self.viewer.mouse_over_canvas
        )

    def _on_mouse_move(self, event: Event) -> None:
        self._last_mouse_pos = event.pos
        self._set_position(event.pos)

    def _set_position(self, pos: tuple[int, int]) -> None:
        if not self.layer._is_resizing_brush:
            self.node.transform.translate = [pos[0], pos[1], 0, 0]

    def _on_canvas_change(self, event: Event) -> None:
        if event.new is not None:
            event.new.events.mouse_move.connect(self._on_mouse_move)
        if event.old is not None:
            event.old.events.mouse_move.disconnect(self._on_mouse_move)

    def _on_mouse_over_canvas(self) -> None:
        if self.viewer.mouse_over_canvas:
            # Move the cursor outside the canvas when the mouse leaves it.
            # It fixes the bug described in PR #5763:
            # https://github.com/napari/napari/pull/5763#issuecomment-1523182141
            self._set_position((-1000, -1000))
            self.node.visible = self.overlay.visible
        else:
            if self.overlay.visible:
                self.node.visible = self.layer._is_resizing_brush
            else:
                self.node.visible = False

    def reset(self) -> None:
        super().reset()
        self._on_size_change()
        self._last_mouse_pos = None

    def close(self) -> None:
        self.node.events.canvas_change.disconnect(self._on_canvas_change)
        super().close()
