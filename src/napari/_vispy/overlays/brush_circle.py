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

        self.overlay.events.size.connect(self._on_size_change)
        self.viewer.cursor.events.canvas_position.connect(self._on_cursor_move)
        # no need to connect position, since that's in the base classes of CanvasOverlay

        self.reset()

    def _on_size_change(self, event: Event | None = None) -> None:
        self._white_circle.radius = self.overlay.size / 2
        self._black_circle.radius = self._white_circle.radius - 1

    def _on_visible_change(self) -> None:
        if self._last_mouse_pos is not None:
            self._set_position(self._last_mouse_pos)
        self.node.visible = self.overlay.visible

    def _on_cursor_move(self) -> None:
        pos = self.viewer.cursor.canvas_position
        if pos is None:
            self._set_position((-1000, -1000))
        else:
            self._set_position(pos)

    def _set_position(self, pos: tuple[int, int]) -> None:
        if not self.overlay.position_is_frozen:
            self.node.transform.translate = [pos[0], pos[1], 0, 0]

    def reset(self) -> None:
        super().reset()
        self._on_size_change()
        self._on_cursor_move()

    def close(self) -> None:
        super().close()
