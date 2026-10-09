from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vispy.scene.visuals import Compound, Ellipse

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay

if TYPE_CHECKING:
    from napari.components.overlays.brush_circle import BrushCircleOverlay
    from napari.utils.events import Event


class VispyBrushCircleOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    overlay: BrushCircleOverlay

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
        self._outside = (10000, 10000)

        self.overlay.events.size.connect(self._on_size_change)
        self.node.events.canvas_change.connect(self._on_canvas_change)
        # no need to connect position, since that's in the base classes of CanvasOverlay

        self.reset()

        # manually connect this once and get the correct canvas
        if self.node.parent is not None:
            self.node.parent.scene.canvas.events.mouse_move.connect(
                self._on_mouse_move
            )

    def _on_mouse_leave(self) -> None:
        self._set_position(self._outside)

    def _on_position_change(self, event: Event | None = None) -> None:
        self._set_position(self.overlay.position)

    def _on_size_change(self, event: Event | None = None) -> None:
        self._white_circle.radius = self.overlay.size / 2
        self._black_circle.radius = self._white_circle.radius - 1

    def _on_mouse_move(self, event: Event) -> None:
        # TODO: this will be replaced by handling the event
        #       in a label mouse callback (after the brush becomes a layer overlay)
        if self.overlay.visible:
            self.overlay.position = event.pos.tolist()

    def _set_position(self, pos: tuple[int, int]) -> None:
        if not self.overlay.position_is_frozen:
            self.node.transform.translate = [pos[0], pos[1], 0, 0]

    def _on_canvas_change(self, event: Event) -> None:
        if event.new is not None:
            event.new.events.mouse_move.connect(self._on_mouse_move)
        if event.old is not None:
            event.old.events.mouse_move.disconnect(self._on_mouse_move)

    def reset(self) -> None:
        super().reset()
        self._on_mouse_leave()
        self._on_size_change()
        self._last_mouse_pos = None

    def close(self) -> None:
        self.node.events.canvas_change.disconnect(self._on_canvas_change)
        super().close()
