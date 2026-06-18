from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.bounding_box import BoundingBox

if TYPE_CHECKING:
    from napari.components.overlays import BoundingBoxOverlay
    from napari.layers._scalar_field import ScalarFieldBase


class VispyBoundingBoxOverlay(LayerOverlayMixin, VispySceneOverlay):
    overlay: BoundingBoxOverlay
    layer: ScalarFieldBase
    node: BoundingBox

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(node=BoundingBox(), **kwargs)
        self.layer.events.set_data.connect(self._on_bounds_change)
        self.overlay.events.lines.connect(self._on_lines_change)
        self.overlay.events.line_thickness.connect(
            self._on_line_thickness_change
        )
        self.overlay.events.line_color.connect(self._on_line_color_change)
        self.overlay.events.points.connect(self._on_points_change)
        self.overlay.events.point_size.connect(self._on_point_size_change)
        self.overlay.events.point_color.connect(self._on_point_color_change)

        self.reset()

    def _on_bounds_change(self) -> None:
        displayed = list(self.layer._slice_input.displayed)

        if self.layer.multiscale:
            # Always use the full level-0 extent so the bounding box
            # represents the complete dataset in both 2D and 3D.
            bounds = self.layer._display_bounding_box_at_level(
                displayed, 0
            ) + np.array([[-0.5, 0.5]])

            # tile2data (transforms[0]) maps tile-local pixels → level-0
            # data coords.  vispy applies tile2data to everything in the
            # node tree, including this overlay.  Transform the level-0
            # bounds through tile2data.inverse so they survive the forward
            # transform and land at the correct world position.
            tile2data = self.layer._transforms[0]
            t2d_scale = np.asarray(getattr(tile2data, 'scale', None))
            t2d_translate = np.asarray(getattr(tile2data, 'translate', None))
            if t2d_scale is not None and t2d_translate is not None:
                disp_scale = t2d_scale[displayed]
                disp_translate = t2d_translate[displayed]
                safe_scale = np.where(
                    np.abs(disp_scale) > 1e-12, disp_scale, 1.0
                )
                bounds = (
                    bounds - disp_translate[:, np.newaxis]
                ) / safe_scale[:, np.newaxis]
        else:
            bounds = self.layer._display_bounding_box_augmented_data_level(
                displayed
            )

        if len(bounds) == 2:
            # 2d layers are assumed to be at 0 in the 3rd dimension
            bounds = np.pad(bounds, ((1, 0), (0, 0)))

        self.node.set_bounds(bounds[::-1])  # invert for vispy

    def _on_lines_change(self) -> None:
        self.node.lines.visible = self.overlay.lines

    def _on_points_change(self) -> None:
        self.node.markers.visible = self.overlay.points

    def _on_line_thickness_change(self) -> None:
        self.node._line_thickness = self.overlay.line_thickness
        self._on_bounds_change()

    def _on_line_color_change(self) -> None:
        self.node._line_color = self.overlay.line_color
        self._on_bounds_change()

    def _on_point_size_change(self) -> None:
        self.node._marker_size = self.overlay.point_size
        self._on_bounds_change()

    def _on_point_color_change(self) -> None:
        self.node._marker_color = self.overlay.point_color
        self._on_bounds_change()

    def reset(self) -> None:
        super().reset()
        self._on_line_thickness_change()
        self._on_line_color_change()
        self._on_point_color_change()
        self._on_point_size_change()
        self._on_points_change()
        self._on_bounds_change()
