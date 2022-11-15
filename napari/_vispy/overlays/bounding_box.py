from napari._vispy.overlays.base import VispyLayerOverlay
from napari._vispy.visuals.bounding_box import BoundingBox


class VispyBoundingBoxOverlay(VispyLayerOverlay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, node=BoundingBox(), **kwargs)
        self.layer.events.set_data.connect(self._on_bounds_change)
        self.layer.events._ndisplay.connect(self._on_bounds_change)
        self.overlay.events.lines.connect(self._on_lines_change)
        self.overlay.events.line_thickness.connect(
            self._on_line_thickness_change
        )
        self.overlay.events.line_color.connect(self._on_line_color_change)
        self.overlay.events.points.connect(self._on_points_change)
        self.overlay.events.point_size.connect(self._on_point_size_change)
        self.overlay.events.point_color.connect(self._on_point_color_change)

    def _on_bounds_change(self):
        bounds = self.layer._display_bounding_box(
            self.layer._slice_input.displayed
        )
        # invert for vispy
        self.node.set_bounds(bounds[::-1])
        self._on_lines_change()

    def _on_lines_change(self):
        if self.layer._slice_input.ndisplay == 2:
            self.node.line2d.visible = self.overlay.lines
            self.node.line3d.visible = False
        else:
            self.node.line3d.visible = self.overlay.lines
            self.node.line2d.visible = False

    def _on_points_change(self):
        self.node.markers.visible = self.overlay.points

    def _on_line_thickness_change(self):
        self.node.line2d.set_data(width=self.overlay.line_thickness)
        self.node.line3d.set_data(width=self.overlay.line_thickness)

    def _on_line_color_change(self):
        self.node.line2d.set_data(color=self.overlay.line_color)
        self.node.line3d.set_data(color=self.overlay.line_color)

    def _on_point_size_change(self):
        self.node._marker_size = self.overlay.point_size
        self._on_bounds_change()

    def _on_point_color_change(self):
        self.node._marker_color = self.overlay.point_color
        self._on_bounds_change()

    def reset(self):
        super().reset()
        self._on_line_thickness_change()
        self._on_line_color_change()
        self._on_point_color_change()
        self._on_point_size_change()
        self._on_points_change()
        self._on_bounds_change()
