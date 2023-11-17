import numpy as np
from vispy.scene.visuals import Compound, Line, Markers, Polygon

from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay
from napari.components.overlays import LabelsPolygonOverlay
from napari.layers import Labels
from napari.layers.labels._labels_constants import Mode
from napari.layers.labels._labels_utils import mouse_event_to_labels_coordinate


def _only_when_enabled(callback):
    """Decorator that wraps a callback of VispyLabelsPolygonOverlay.

    It ensures that the callback is only executed when all the conditions are met:
    1) The overlay is enabled;
    2) The number of displayed dimensions is 2 (it can only work in 2D);
    3) The number of dimensions across which labels will be edited is 2.

    If 2, 3 are not met, the Labels mode is automatically switched to PAN_ZOOM.
    """

    def decorated_callback(self, layer: Labels, event):
        if not self.overlay.enabled:
            return

        if layer._slice_input.ndisplay != 2 or layer.n_edit_dimensions != 2:
            layer.mode = Mode.PAN_ZOOM
            return
        callback(self, layer, event)

    return decorated_callback


class VispyLabelsPolygonOverlay(LayerOverlayMixin, VispySceneOverlay):
    layer: Labels

    def __init__(
        self, *, layer: Labels, overlay: LabelsPolygonOverlay, parent=None
    ):
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

        self.layer.mouse_move_callbacks.append(self._on_mouse_move)
        self.layer.mouse_drag_callbacks.append(self._on_mouse_press)
        self.layer.mouse_double_click_callbacks.append(
            self._on_mouse_double_click
        )

        self.overlay.events.points.connect(self._on_points_change)
        self.overlay.events.enabled.connect(self._on_enabled_change)

        layer.events.selected_label.connect(self._update_color)
        layer.events.colormap.connect(self._update_color)
        layer.events.color_mode.connect(self._update_color)
        layer.events.opacity.connect(self._update_color)

        self._first_point_pos = np.zeros(2)

        self.reset()
        self._update_color()
        # If there are no points, it won't be visible
        self.overlay.visible = True

    def _on_enabled_change(self):
        if self.overlay.enabled:
            self._on_points_change()

    def _on_points_change(self):
        num_points = len(self.overlay.points)
        if num_points:
            points = np.array(self.overlay.points)[
                :, self._dims_displayed[::-1]
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
        if layer._selected_label == layer._background_label:
            self._set_color((1, 0, 0, 0))
        else:
            self._set_color(
                layer._selected_color.tolist()[:3] + [layer.opacity]
            )

    @_only_when_enabled
    def _on_mouse_move(self, layer, event):
        """Continuously redraw the latest polygon point with the current mouse position."""
        if self._num_points == 0:
            return

        pos = self._get_mouse_coordinates(event)
        self.overlay.points = self.overlay.points[:-1] + [pos.tolist()]

    @_only_when_enabled
    def _on_mouse_press(self, layer, event):
        pos = self._get_mouse_coordinates(event)
        dims_displayed = self._dims_displayed

        if event.button == 1:  # left mouse click
            orig_pos = pos.copy()
            # recenter the point in the center of the image pixel
            pos[dims_displayed] = np.floor(pos[dims_displayed]) + 0.5

            if not self.overlay.points:
                self._first_point_pos = np.array(event.pos)

            prev_point = (
                self.overlay.points[-2] if self._num_points > 1 else None
            )
            # Add a new point only if it differs from the previous one
            if prev_point is None or np.linalg.norm(pos - prev_point) > 0:
                self.overlay.points = self.overlay.points[:-1] + [
                    pos.tolist(),
                    # add some epsilon to avoid points duplication,
                    # the latest point is used only for visualization of the cursor
                    (orig_pos + 1e-3).tolist(),
                ]
        elif event.button == 2 and self._num_points > 0:  # right mouse click
            if self._num_points < 3:
                self.overlay.points = []
            else:
                self.overlay.points = self.overlay.points[:-2] + [pos.tolist()]

    @_only_when_enabled
    def _on_mouse_double_click(self, layer, event):
        if event.button == 2:
            self._on_mouse_press(layer, event)
            return None

        first_point_dist = np.linalg.norm(event.pos - self._first_point_pos)
        if (
            not self.overlay.double_click_completion
            or first_point_dist > self.overlay.completion_radius
        ):
            return self._on_mouse_press(layer, event)

        # Remove the latest 2 points as double click always follows a simple click
        # and another point is reserved for the visualization purpose
        self.overlay.points = self.overlay.points[:-2]
        self.overlay.add_polygon_to_labels(layer)
        return None

    def _get_mouse_coordinates(self, event):
        pos = mouse_event_to_labels_coordinate(self.layer, event)
        if pos is None:
            return None

        pos = np.array(pos, dtype=float)
        pos[self._dims_displayed] += 0.5

        return pos

    @property
    def _dims_displayed(self):
        return self.layer._slice_input.displayed

    @property
    def _num_points(self):
        return len(self.overlay.points)

    def reset(self):
        super().reset()
        self._on_points_change()
