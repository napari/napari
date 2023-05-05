import numpy as np
from vispy.scene.visuals import Compound, Line, Markers, Polygon

from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay


class VispyLabelsPolygonOverlay(LayerOverlayMixin, VispySceneOverlay):
    def __init__(self, *, layer, overlay, parent=None):
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

        super().__init__(
            node=Compound([self._polygon, self._nodes, self._line]),
            layer=layer,
            overlay=overlay,
            parent=parent,
        )
        self.overlay.events.points.connect(self._on_points_change)
        self.overlay.events.color.connect(self._on_color_change)
        self.overlay.events.dims_order.connect(self._on_points_change)

        layer.events.selected_label.connect(self._update_color)
        layer.events.colormap.connect(self._update_color)
        layer.events.color_mode.connect(self._update_color)
        layer.events.opacity.connect(self._update_color)

        self.reset()

    def _on_points_change(self):
        num_points = len(self.overlay.points)
        if num_points:
            points = np.array(self.overlay.points)[
                :, self.overlay.dims_order[::-1]
            ]
        else:
            points = np.empty((0, 2))

        if num_points > 2:
            self._polygon.visible = True
            self._line.visible = False
            self._polygon.pos = points
        else:
            self._polygon.visible = False
            self._line.visible = num_points == 2
            if self._line.visible:
                self._line.set_data(pos=points)

        self._nodes.set_data(
            pos=points,
            **self._nodes_kwargs,
        )

    def _on_color_change(self):
        border_color = self.overlay.color[:3] + (1,)  # always opaque

        polygon_color = self.overlay.color

        # Workaround for VisPy's polygon bug: if you set opacity to exactly 0,
        # it keeps the previous visualization of the polygon without cleaning it
        make_transparent = (
            not self._polygon.color.is_blank and polygon_color[-1] == 0
        )
        # Temporarily set a degenerate polygon to clean its faces
        if make_transparent:
            self._polygon.pos = [(0, 0), (1, 1)]

        self._polygon.color = polygon_color

        # Restore the original polygon when it is transparent
        if make_transparent:
            self._on_points_change()

        self._polygon.border_color = border_color
        self._line.set_data(color=border_color)

    def _update_color(self):
        layer = self.layer
        if layer._selected_label == layer._background_label:
            self.overlay.color = (1, 0, 0, 0)
        else:
            self.overlay.color = layer._selected_color.tolist()[:3] + [
                layer.opacity
            ]

    def reset(self):
        super().reset()
        self._on_points_change()
        self._update_color()
