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
        self._outside = (-1000, -1000)
        self._last_mouse_pos = self._outside

        self.layer.events.brush_size.connect(self._on_size_change)
        self.layer.events.brush_size_is_canvas_pixels.connect(
            self._on_size_change
        )
        self.viewer.scene.camera.events.zoom.connect(self._on_size_change)
        self.viewer.cursor.events.canvas_position.connect(
            self._on_canvas_position_change
        )

        self.reset()

    def _on_position_change(self, event: Event | None = None) -> None:
        # TODO: this overrides behaviuour of tiled overlays. To be removed
        #       with #9083
        pass

    def _on_canvas_position_change(self) -> None:
        if self.overlay._is_resizing:
            return

        pos = self.viewer.cursor.canvas_position
        self._set_position(pos[::-1] if pos is not None else self._outside)

    def _set_position(self, pos: tuple[int, int]) -> None:
        self.node.transform.translate = [pos[0], pos[1], 0, 0]
        self.node.visible = True
        self._last_mouse_pos = pos

    def _on_size_change(self, event: Event | None = None) -> None:
        size = self.layer._get_brush_size_canvas(self.viewer.scene.camera.zoom)
        self._white_circle.radius = size / 2
        self._black_circle.radius = self._white_circle.radius - 1

    def _on_visible_change(self) -> None:
        self._set_position(self._last_mouse_pos)
        self.node.visible = self.overlay.visible

    def reset(self) -> None:
        super().reset()
        self._on_size_change()
        self._on_canvas_position_change()
