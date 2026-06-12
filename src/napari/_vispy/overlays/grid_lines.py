from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.grid_lines import GridLines3D
from napari.components.camera import DEFAULT_ORIENTATION_TYPED
from napari.settings import get_settings

if TYPE_CHECKING:
    from vispy.scene import Node
    from vispy.visuals.text.text import FontManager

    from napari.components.overlays import GridLinesOverlay, Overlay
    from napari.components.viewer_model import ViewerModel


class VispyGridLinesOverlay(ViewerOverlayMixin, VispySceneOverlay):
    overlay: GridLinesOverlay
    node: GridLines3D

    def __init__(
        self,
        *,
        viewer: ViewerModel,
        overlay: Overlay,
        parent: Node = None,
        font_manager: FontManager | None = None,
        font_family: str = 'OpenSans',
    ):
        super().__init__(
            node=GridLines3D(
                font_manager=font_manager, font_family=font_family
            ),
            viewer=viewer,
            overlay=overlay,
            parent=parent,
        )

        self.overlay.events.color.connect(self._rebuild_all)
        self.overlay.events.axis_labels.connect(self._on_axis_labels_change)
        self.overlay.events.tick_labels.connect(self._on_ticks_change)
        self.overlay.events.n_ticks.connect(self._on_ticks_change)
        self.viewer.dims.events.order.connect(self._rebuild_all)
        self.viewer.dims.events.range.connect(self._on_extent_change)
        self.viewer.dims.events.ndisplay.connect(self._rebuild_all)
        self.viewer.dims.events.axis_labels.connect(
            self._on_axis_labels_change
        )

        # would be nice to fire this less often to save performance
        self.viewer.camera.events.angles.connect(
            self._on_view_direction_change
        )
        self.viewer.camera.events.orientation.connect(
            self._on_view_direction_change
        )
        self.viewer.camera.events.zoom.connect(self._on_view_direction_change)
        get_settings().appearance.events.theme.connect(self._rebuild_all)
        self.viewer.events.theme.connect(self._rebuild_all)

        self.reset()

    def _get_ranges_and_axis_labels(self):
        # napari dims are zyx, but vispy uses xyz
        displayed = self.viewer.dims.displayed[::-1]
        ranges = [self.viewer.dims.range[i] for i in displayed]
        axis_labels = [self.viewer.dims.axis_labels[i] for i in displayed]
        return ranges, axis_labels

    def _on_axis_labels_change(self):
        ranges, axis_labels = self._get_ranges_and_axis_labels()
        self.node.set_axis_labels(
            self.overlay.axis_labels, ranges, axis_labels
        )
        self._on_blending_change()  # needed to ensure new grids/ticks are up to date

    def _on_ticks_change(self):
        ranges, _ = self._get_ranges_and_axis_labels()
        self.node.set_ticks(
            self.overlay.tick_labels, self.overlay.n_ticks, ranges
        )
        self._on_blending_change()  # needed to ensure new grids/ticks are up to date

    def _on_extent_change(self):
        ranges, _ = self._get_ranges_and_axis_labels()
        self.node.set_extents(ranges)
        self._on_ticks_change()
        self._on_axis_labels_change()

    def _rebuild_all(self) -> None:
        self.node.color = (
            self.overlay.color
            if self.overlay.color is not None
            else self._get_fgcolor()
        )
        self.node.reset_grids()
        self._on_extent_change()

        self._on_view_direction_change()

    def _on_view_direction_change(self) -> None:
        # all is flipped from zyx to xyz for vispy
        displayed = self.viewer.dims.displayed[::-1]
        ranges = tuple(self.viewer.dims.range[i] for i in displayed)

        if self.viewer.dims.ndisplay == 3:
            view_direction = np.sign(self.viewer.camera.view_direction)
            orientation_flip = tuple(
                1 if ori == default_ori else -1
                for ori, default_ori in zip(
                    self.viewer.camera.orientation,
                    DEFAULT_ORIENTATION_TYPED,
                    strict=True,
                )
            )

            view_is_flipped = tuple((view_direction * orientation_flip) >= 0)[
                ::-1
            ]
        else:
            view_is_flipped = (False, False, False)
            ranges = ranges + ((0, 0),)

        self.node.set_view_direction(
            ranges,
            view_is_flipped,
            zoom=self.viewer.camera.zoom,
        )

    def reset(self) -> None:
        super().reset()
        self._rebuild_all()
