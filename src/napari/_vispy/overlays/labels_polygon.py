from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vispy.scene.visuals import Compound, Ellipse, Line, Markers, Polygon

from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay
from napari.settings import get_settings

if TYPE_CHECKING:
    from napari.components.overlays import LabelsPolygonOverlay
    from napari.layers import Labels


class VispyLabelsPolygonOverlay(LayerOverlayMixin, VispySceneOverlay):
    layer: Labels
    overlay: LabelsPolygonOverlay

    def __init__(self, **kwargs):
        points = [(0, 0), (1, 1)]

        self._nodes_kwargs = {
            'face_color': (1, 1, 1, 1),
            'size': 8.0,
            'edge_width': 1.0,
            'edge_color': (0, 0, 0, 1),
        }

        self._nodes = Markers(pos=np.array(points), **self._nodes_kwargs)

        self._polygon = Polygon(
            pos=points,
            border_method='agg',
        )

        self._line = Line(pos=points, method='agg')

        self._circle = Ellipse(
            center=(0, 0),
            radius=1.0,
            color=(0, 0, 0, 0),  # transparent fill
            border_color=(1, 0, 0, 1),  # red outline
            border_method='agg',
        )
        self._show_circle = False

        super().__init__(
            node=Compound(
                [self._polygon, self._nodes, self._line, self._circle]
            ),
            **kwargs,
        )

        self.overlay.events.points.connect(self._on_points_change)
        self.overlay.events.floating_point.connect(self._on_points_change)

        self.layer.events.selected_label.connect(self._update_color)
        self.layer.events.colormap.connect(self._update_color)
        self.layer.events.opacity.connect(self._update_color)

        # set completion radius based on settings
        self._on_completion_radius_change()
        get_settings().experimental.events.completion_radius.connect(
            self._on_completion_radius_change
        )

        self.reset()
        self._update_color()

    def _on_completion_radius_change(self, event=None):
        completion_radius = get_settings().experimental.completion_radius
        self._show_circle = completion_radius > 0
        if completion_radius > 0:
            self._circle.radius = completion_radius

    def _on_points_change(self):
        points = list(self.overlay.points)
        if points:
            # Create full-dimensional points for transformation
            if self.overlay.floating_point is not None:
                points.append(self.overlay.floating_point)
            points_full = np.array(points)

            # Apply tile2data inverse transform if downsampling is active.
            # Polygon points are stored in data coordinates, but the layer's
            # vispy visual uses texture coordinates when downsampling is active.
            # We need to convert from data space to texture space for correct rendering.
            tile2data = self.layer._transforms['tile2data']
            if hasattr(tile2data, 'scale') and not np.allclose(
                tile2data.scale, 1.0
            ):
                points_full = tile2data.inverse(points_full)

            points = points_full[:, self.layer._slice_input.displayed[::-1]]
        else:
            points = np.empty((0, 2))

        if len(points):
            self._circle.center = points[0]
            self._circle.visible = self._show_circle
        else:
            self._circle.visible = False

        if len(points) > 2:
            self._polygon.visible = True
            self._line.visible = False
            self._polygon.pos = points
        else:
            self._polygon.visible = False
            self._line.visible = len(points) == 2
            if self._line.visible:
                self._line.set_data(pos=points)

        self._nodes.set_data(
            pos=points,
            **self._nodes_kwargs,
        )

    def _set_color(self, color):
        border_color = tuple(color[:3]) + (1,)  # always opaque
        polygon_color = color

        # Clean up polygon faces before making it transparent, otherwise
        # it keeps the previous visualization of the polygon without cleaning
        if polygon_color[-1] == 0:
            self._polygon.mesh.set_data(faces=[])
        self._polygon.color = polygon_color

        self._polygon.border_color = border_color
        self._line.set_data(color=border_color)

    def _update_color(self):
        layer = self.layer
        if layer._selected_label == layer.colormap.background_value:
            self._set_color((1, 0, 0, 0))
        else:
            self._set_color(
                layer._selected_color.tolist()[:3] + [layer.opacity]  # pyrefly: ignore [missing-attribute]
            )

    def reset(self):
        super().reset()
        self._on_points_change()

    def close(self):
        get_settings().experimental.events.completion_radius.disconnect(
            self._on_completion_radius_change
        )
        super().close()
