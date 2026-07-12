from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vispy.scene.visuals import Compound, Ellipse

from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay
from napari.layers.labels._labels_utils import mouse_event_to_labels_coordinate

if TYPE_CHECKING:
    from napari.components.overlays import LabelsBrushStrokeOverlay
    from napari.layers import Labels

# circle radius (data units) as a multiple of the brush size
RADIUS_FACTOR = 1.8


def _only_when_overlay_enabled(callback):
    """Run the callback only when the overlay is enabled and editing in 2D.

    Unlike the labels polygon overlay (which is its own mode), this overlay is
    active during PAINT mode, which also supports 3D painting. So when editing
    is not 2D we silently no-op rather than switching mode, leaving the normal
    brush untouched.
    """

    def decorated_callback(self, layer: Labels, event):
        if not self.overlay.enabled:
            return None
        if layer._slice_input.ndisplay != 2 or layer.n_edit_dimensions != 2:
            return None
        # returns a generator for the press callback (pan-during-stroke)
        return callback(self, layer, event)

    return decorated_callback


class VispyLabelsBrushStrokeOverlay(LayerOverlayMixin, VispySceneOverlay):
    layer: Labels
    overlay: LabelsBrushStrokeOverlay

    def __init__(self, **kwargs):
        self._circle = Ellipse(
            center=(0, 0),
            radius=1.0,
            color=(0, 0, 0, 0),  # transparent fill
            border_color=(1, 0, 0, 1),  # red outline
            border_method='agg',
        )

        super().__init__(node=Compound([self._circle]), **kwargs)

        # transient interaction state (not rendered, so kept off the model)
        self._stroke_points: list = []
        self._last_coord = None
        self._has_left = False

        self.layer.mouse_move_callbacks.append(self._on_mouse_move)
        self.layer.mouse_drag_callbacks.append(self._on_mouse_press)

        self.overlay.events.active.connect(self._on_geometry_change)
        self.overlay.events.center.connect(self._on_geometry_change)
        self.overlay.events.radius.connect(self._on_geometry_change)
        self.overlay.events.enabled.connect(self._on_enabled_change)

        self.reset()

    def _on_enabled_change(self):
        # abort a stroke if the tool is disabled mid-stroke (e.g. mode switch)
        if not self.overlay.enabled and self.overlay.active:
            self.overlay.abort(self.layer)

    def _on_geometry_change(self, event=None):
        if not self.overlay.active:
            self._circle.visible = False
            return

        dd = list(self._dims_displayed)
        center = np.array(self.overlay.center, dtype=float)
        # radius can get recast to a tuple below when scale differs per axis
        radius: float | tuple[float, float] = float(self.overlay.radius)

        # convert data -> texture space when downsampling is active, like the
        # labels polygon overlay does for its points
        tile2data = self.layer._transforms['tile2data']
        if hasattr(tile2data, 'scale') and not np.allclose(
            tile2data.scale, 1.0
        ):
            center = np.asarray(tile2data.inverse(center), dtype=float)
            scale = np.abs(np.asarray(tile2data.scale))[dd]
            radius = (radius / scale[1], radius / scale[0])

        self._circle.center = tuple(center[dd][::-1])
        self._circle.radius = radius
        self._circle.visible = True

    @_only_when_overlay_enabled
    def _on_mouse_press(self, layer, event):
        if not self.overlay.active:
            # the right button starts an encircle-and-fill stroke
            if event.button == 2:
                self._start_stroke(layer, event)
            return
        if event.button == 2:
            return
        # a press during an encircle-stroke pans the canvas until release, then the
        # stroke resumes on the next mouse move
        previous_mouse_pan = layer.mouse_pan
        layer.mouse_pan = True
        yield
        while event.type == 'mouse_move':
            yield
        layer.mouse_pan = previous_mouse_pan
        # avoid interpolating a stray line across the panned gap
        self._last_coord = None

    def _start_stroke(self, layer, event):
        coord = mouse_event_to_labels_coordinate(layer, event)
        if coord is None:
            return
        self._reset_stroke_state()
        # hold history open across all the discrete events of this stroke
        layer._begin_stroke()
        self.overlay.center = tuple(coord)
        self.overlay.radius = layer.brush_size * RADIUS_FACTOR
        self.overlay.active = True
        self._paint_to(layer, coord)

    @_only_when_overlay_enabled
    def _on_mouse_move(self, layer, event):
        if not self.overlay.active:
            return
        coord = mouse_event_to_labels_coordinate(layer, event)
        if coord is None:
            return
        self._paint_to(layer, coord)
        if self._is_at_start(layer, coord):
            self._complete(layer)

    def _paint_to(self, layer, coord):
        last = self._last_coord if self._last_coord is not None else coord
        layer._draw(layer.selected_label, last, coord)
        self._last_coord = coord
        self._stroke_points.append(list(coord))

    def _is_at_start(self, layer, coord) -> bool:
        dd = list(self._dims_displayed)
        center = np.array(self.overlay.center, dtype=float)
        dist = np.linalg.norm(np.array(coord, dtype=float)[dd] - center[dd])
        radius = self.overlay.radius
        # require the cursor to leave the radius before a return can complete,
        # otherwise the stroke would complete instantly at the start point
        if not self._has_left:
            if dist > radius:
                self._has_left = True
            return False
        return dist <= radius

    def _complete(self, layer):
        # continue the stroke all the way back to the start point so the drawn
        # outline closes cleanly instead of leaving a notch where the cursor
        # re-entered the stop radius
        self._paint_to(layer, np.array(self.overlay.center, dtype=float))
        if len(self._stroke_points) > 2:
            layer.paint_polygon(self._stroke_points, layer.selected_label)
        layer._commit_stroke()
        self.overlay.active = False  # hides the circle

    def _reset_stroke_state(self):
        self._stroke_points = []
        self._last_coord = None
        self._has_left = False

    @property
    def _dims_displayed(self):
        return self.layer._slice_input.displayed

    def reset(self):
        super().reset()
        self._on_geometry_change()

    def close(self):
        self.layer.mouse_move_callbacks.remove(self._on_mouse_move)
        self.layer.mouse_drag_callbacks.remove(self._on_mouse_press)
        super().close()
