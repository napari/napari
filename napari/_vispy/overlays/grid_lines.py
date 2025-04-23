import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.grid_lines import GridLines3D


class VispyGridLinesOverlay(ViewerOverlayMixin, VispySceneOverlay):
    def __init__(self, *, viewer, overlay, parent=None):
        super().__init__(
            node=GridLines3D(),
            viewer=viewer,
            overlay=overlay,
            parent=parent,
        )

        self._old_directions = [1, 1, 1]

        self.overlay.events.ticks.connect(self._on_ticks_change)
        self.overlay.events.tick_spacing.connect(self._on_ticks_change)
        self.viewer.dims.events.order.connect(self._on_data_change)
        self.viewer.dims.events.range.connect(self._on_data_change)
        self.viewer.dims.events.ndisplay.connect(self._on_data_change)
        self.viewer.camera.events.angles.connect(
            self._on_view_direction_change
        )

        self.reset()

    def _on_data_change(self):
        # napari dims are zyx, but vispy uses xyz
        displayed = self.viewer.dims.displayed[::-1]
        ranges = [self.viewer.dims.range[i] for i in displayed]

        self.node.set_extents(displayed, ranges)
        self._on_ticks_change()

    def _on_view_direction_change(self):
        # this works but it's EXPENSIVE, we need to debounce or find a cheaper check
        directions = np.array(self.viewer.camera.view_direction)[::-1] >= 0
        if np.array_equal(self._old_directions, directions):
            return

        # flip to xyz
        displayed = self.viewer.dims.displayed[::-1]
        ranges = [self.viewer.dims.range[i] for i in displayed]

        self.node.set_view_direction(
            ranges,
            directions,
        )
        self._old_directions = directions

    def _on_ticks_change(self):
        # flip to xyz
        displayed = self.viewer.dims.displayed[::-1]
        ranges = [self.viewer.dims.range[i] for i in displayed]
        self.node.set_ticks(
            self.overlay.ticks, self.overlay.tick_spacing, ranges
        )
        self._on_view_direction_change()

    def reset(self):
        super().reset()
        self._on_data_change()
