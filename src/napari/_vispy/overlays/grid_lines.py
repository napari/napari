from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.grid_lines import GridLines3D
from napari.components.dims import RangeTuple
from napari.settings import get_settings

if TYPE_CHECKING:
    from napari._vispy.utils.qt_font import FontInfo
    from napari.components.overlays import GridLinesOverlay


class VispyGridLinesOverlay(ViewerOverlayMixin, VispySceneOverlay):
    overlay: GridLinesOverlay
    node: GridLines3D

    def __init__(
        self,
        *,
        font_info: FontInfo,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node=GridLines3D(
                font_info=font_info,
            ),
            font_info=font_info,
            **kwargs,
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
        self.viewer.scene.camera.events.angles.connect(
            self._on_view_direction_change
        )
        self.viewer.scene.camera.events.orientation.connect(
            self._on_view_direction_change
        )
        self.viewer.scene.camera.events.zoom.connect(
            self._on_view_direction_change
        )
        get_settings().appearance.events.theme.connect(self._rebuild_all)
        self.viewer.events.theme.connect(self._rebuild_all)

        self.reset()

    def _get_ranges_and_axis_labels(
        self,
    ) -> tuple[list[RangeTuple], list[str]]:
        # napari dims are zyx, but vispy uses xyz
        displayed = self.viewer.dims.displayed[::-1]
        ranges = [self.viewer.dims.range[i] for i in displayed]
        axis_labels = [self.viewer.dims.axis_labels[i] for i in displayed]

        if self.viewer.layers.units is not None:
            # see https://pint.readthedocs.io/en/stable/user/formatting.html
            units = [
                f' ({self.viewer.layers.units[i]:~#P})' for i in displayed
            ]
        else:
            units = ['' for _ in displayed]

        axis_labels_with_units = [
            f'{lab}{unit}'
            for lab, unit in zip(axis_labels, units, strict=True)
        ]
        return ranges, axis_labels_with_units

    def _on_unit_change(self):
        # NOTE: this is also called by VispyCanvas when layer units are updated
        #       so it doesn't need to be connected to events for that
        self._on_axis_labels_change()

    def _on_axis_labels_change(self) -> None:
        ranges, axis_labels = self._get_ranges_and_axis_labels()
        self.node.set_axis_labels(
            self.overlay.axis_labels, ranges, axis_labels
        )
        # needed to ensure new grids/ticks are up to date
        self._on_blending_change()
        self._on_view_direction_change()

    def _on_ticks_change(self) -> None:
        ranges, _ = self._get_ranges_and_axis_labels()
        self.node.set_ticks(
            self.overlay.tick_labels, self.overlay.n_ticks, ranges
        )
        # needed to ensure new grids/ticks are up to date
        self._on_blending_change()
        self._on_view_direction_change()

    def _on_extent_change(self) -> None:
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

        if len(ranges) == 3:
            view_direction = np.sign(self.viewer.scene.camera.view_direction)
            view_is_flipped = tuple(view_direction >= 0)[::-1]
        elif len(ranges) == 2:
            view_is_flipped = (False, False, False)
            ranges = ranges + (RangeTuple(0, 0, 1),)
        else:
            raise RuntimeError('unreachable')

        self.node.set_view_direction(
            ranges,
            view_is_flipped,
            zoom=self.viewer.scene.camera.zoom,
        )

    def reset(self) -> None:
        super().reset()
        self._rebuild_all()
