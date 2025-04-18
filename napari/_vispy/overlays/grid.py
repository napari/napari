import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.grid import Grid


class VispyGridOverlay(ViewerOverlayMixin, VispySceneOverlay):
    def __init__(self, *, viewer, overlay, parent=None):
        super().__init__(
            node=Grid(),
            viewer=viewer,
            overlay=overlay,
            parent=parent,
        )

        self._old_directions = [1, 1, 1]

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
        ndim = len(displayed)
        ranges = []
        for i in range(ndim):
            dim0 = displayed[i]
            dim1 = displayed[(i + 1) % ndim]
            rng0 = self.viewer.dims.range[dim0]
            rng1 = self.viewer.dims.range[dim1]
            ranges.append((rng0.start, rng0.stop, rng1.start, rng1.stop))
        if ndim == 2:
            ranges.append(None)

        self.node.set_extents(ranges)

    def _on_view_direction_change(self):
        if self.viewer.dims.ndisplay == 2:
            return

        # this works but it's EXPENSIVE, we need to debounce or find a cheaper check
        directions = np.sign(self.viewer.camera.view_direction)[::-1]
        if np.array_equal(self._old_directions, directions):
            return

        # flip to xyz
        displayed = self.viewer.dims.order[-3:][::-1]  # so it's always 3
        top_bounds = [self.viewer.dims.range[i].stop for i in displayed]

        self.node.set_view_directions(
            top_bounds,
            directions,
        )
        self._old_directions = directions

    def reset(self):
        super().reset()
        self._on_data_change()
        self._on_view_direction_change()
