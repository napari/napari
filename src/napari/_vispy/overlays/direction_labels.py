from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vispy.scene.visuals import Compound

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.text import Text
from napari.components._direction_edge_labels import (
    direction_edge_labels,
    reconcile_direction_labels,
)
from napari.settings import get_settings

if TYPE_CHECKING:
    from napari._vispy.utils.qt_font import FontInfo
    from napari.components.overlays import DirectionLabelsOverlay
    from napari.utils.events import Event

# Gap in canvas pixels between each label and its edge.
_EDGE_MARGIN = 8

# Each label is center-anchored and its *center* is inset from the edge by the
# margin plus half the label's own (width, height), so the whole glyph clears
# the edge at any font size. vispy's directional anchors place text
# inconsistently across the vertical edges, so we anchor uniformly at the center
# and compute the inset ourselves. Canvas-overlay space is pixels, origin
# top-left. Each function maps (canvas w, h, text tw, th) -> center (x, y).
_EDGE_POS = {
    'top': lambda w, h, tw, th: (w / 2, _EDGE_MARGIN + th / 2),
    'bottom': lambda w, h, tw, th: (w / 2, h - _EDGE_MARGIN - th / 2),
    'left': lambda w, h, tw, th: (_EDGE_MARGIN + tw / 2, h / 2),
    'right': lambda w, h, tw, th: (w - _EDGE_MARGIN - tw / 2, h / 2),
}


def _node_canvas(node):
    """The vispy canvas a node is parented into, or None."""
    parent = node.parent
    return parent.scene.canvas if parent is not None else None


class VispyDirectionLabelsOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    """Direction labels pinned to the four edges of a 2D canvas.

    A free (non-tiled) canvas overlay holding one ``Text`` per edge. On camera,
    dims, or label changes it recomputes which label faces each edge via
    ``direction_edge_labels`` (over the overlay's labels reconciled to the
    current ``ndim``) and shows/hides each edge's text; nothing is shown in 3D or
    a degenerate view. Positions are refreshed on canvas resize.
    """

    overlay: DirectionLabelsOverlay

    def __init__(self, font_info: FontInfo, **kwargs: Any) -> None:
        self._texts: dict[str, Text] = {}
        for edge in _EDGE_POS:
            self._texts[edge] = Text(
                font_info=font_info, anchor_x='center', anchor_y='center'
            )
        super().__init__(
            node=Compound(list(self._texts.values())),
            font_info=font_info,
            **kwargs,
        )

        # Recompute when the labels, the displayed axes, or the camera
        # orientation change; refresh the default (canvas-contrasting) color on a
        # theme change (box=False makes the base's theme->box handler a no-op for
        # us, so we subscribe to theme directly like the text overlay). Held as a
        # list so close() can disconnect them symmetrically.
        self._connections = [
            (self.overlay.events.labels, self._on_labels_change),
            (self.overlay.events.color, self._on_color_change),
            (self.overlay.events.font_size, self._on_font_size_change),
            (self.viewer.dims.events.ndim, self._on_labels_change),
            (self.viewer.dims.events.order, self._on_labels_change),
            (self.viewer.dims.events.ndisplay, self._on_labels_change),
            (self.viewer.camera.events.orientation, self._on_labels_change),
            (self.viewer.camera.events.orientation2d, self._on_labels_change),
            (self.viewer.events.theme, self._on_color_change),
            (get_settings().appearance.events.theme, self._on_color_change),
        ]
        for emitter, callback in self._connections:
            emitter.connect(callback)

        # Reposition on canvas resize. The base __init__ has already parented the
        # node, so the initial canvas_change won't reach us; connect the current
        # canvas explicitly (as the brush-circle overlay does) and track it so
        # reparenting and close() can disconnect.
        self._resize_canvas = None
        self.node.events.canvas_change.connect(self._on_canvas_change)

        self.reset()
        self._connect_resize_canvas(_node_canvas(self.node))

    def _canvas_wh(self) -> tuple[float, float]:
        """Canvas (width, height) in pixels.

        Read straight from the vispy canvas when parented, so positions reflect
        the *actual* size at render time. viewer._canvas_size is updated by a
        separate resize handler with no ordering guarantee relative to ours, so
        it is only a fallback (headless / not yet parented). viewer._canvas_size
        is (height, width); a vispy SceneCanvas.size is (width, height).
        """
        if self._resize_canvas is not None:
            width, height = self._resize_canvas.size
            return float(width), float(height)
        height, width = self.viewer._canvas_size
        return float(width), float(height)

    def _current_edges(self) -> dict[str, str] | None:
        """Which label faces each edge now, or None when undefined (3D)."""
        labels = reconcile_direction_labels(
            self.overlay.labels, self.viewer.dims.ndim
        )
        return direction_edge_labels(
            labels, dims=self.viewer.dims, camera=self.viewer.camera
        )

    def _on_labels_change(self, event: Event | None = None) -> None:
        edges = self._current_edges() or {}
        for edge, text in self._texts.items():
            label = edges.get(edge)
            if label is None:
                text.visible = False
            else:
                text.text = label
                text.visible = True
        self._reposition()

    def _reposition(self) -> None:
        width, height = self._canvas_wh()
        # get_width_height needs a live font/GL context and aborts without one
        # (headless, or before the node is parented). Use a zero inset there --
        # nothing is rendered to clip in that case anyway.
        have_context = self._resize_canvas is not None
        for edge, text in self._texts.items():
            tw, th = text.get_width_height() if have_context else (0.0, 0.0)
            text.pos = _EDGE_POS[edge](width, height, tw, th)

    def _on_color_change(self, event: Event | None = None) -> None:
        color = (
            self.overlay.color
            if self.overlay.color is not None
            else self._get_fgcolor()
        )
        for text in self._texts.values():
            text.color = color

    def _on_font_size_change(self, event: Event | None = None) -> None:
        for text in self._texts.values():
            text.font_size = self.overlay.font_size
        self._reposition()

    def _on_position_change(self, event: Event | None = None) -> None:
        # Free overlay: it owns all four edges, so ignore the tiling machinery
        # and reposition directly against the canvas size.
        self._reposition()

    def _connect_resize_canvas(self, canvas) -> None:
        """Subscribe the given canvas's resize (disconnecting any previous)."""
        if canvas is self._resize_canvas:
            return
        if self._resize_canvas is not None:
            self._resize_canvas.events.resize.disconnect(self._on_resize)
        self._resize_canvas = canvas
        if canvas is not None:
            canvas.events.resize.connect(self._on_resize)
            self._reposition()

    def _on_canvas_change(self, event: Event | None = None) -> None:
        self._connect_resize_canvas(_node_canvas(self.node))

    def _on_resize(self, event: Event | None = None) -> None:
        self._reposition()

    def reset(self) -> None:
        super().reset()
        self._on_font_size_change()
        self._on_color_change()
        self._on_labels_change()

    def close(self) -> None:
        self._connect_resize_canvas(None)
        self.node.events.canvas_change.disconnect(self._on_canvas_change)
        for emitter, callback in self._connections:
            emitter.disconnect(callback)
        super().close()
