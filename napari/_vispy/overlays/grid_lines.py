import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.grid_lines import GridLines3D
from napari.components.camera import DEFAULT_ORIENTATION_TYPED


class VispyGridLinesOverlay(ViewerOverlayMixin, VispySceneOverlay):
    def __init__(self, *, viewer, overlay, parent=None):
        super().__init__(
            node=GridLines3D(),
            viewer=viewer,
            overlay=overlay,
            parent=parent,
        )

        self.overlay.events.color.connect(self._on_data_change)
        self.overlay.events.tick_labels.connect(self._on_data_change)
        self.overlay.events.n_ticks.connect(self._on_data_change)
        self.viewer.dims.events.order.connect(self._on_data_change)
        self.viewer.dims.events.range.connect(self._on_data_change)
        self.viewer.dims.events.ndisplay.connect(self._on_data_change)
        self.viewer.dims.events.axis_labels.connect(self._on_data_change)

        # would be nice to fire this less often to save performance
        self.viewer.camera.events.angles.connect(
            self._on_view_direction_change
        )
        self.viewer.camera.events.orientation.connect(
            self._on_view_direction_change
        )

        self.reset()

    def _on_data_change(self):
        # napari dims are zyx, but vispy uses xyz
        displayed = self.viewer.dims.displayed[::-1]
        ranges = [self.viewer.dims.range[i] for i in displayed]

        self.node.reset_grids(self.overlay.color)
        self.node.set_extents(ranges)
        self.node.set_ticks(
            self.overlay.tick_labels, self.overlay.n_ticks, ranges
        )

        self._on_view_direction_change(force=True)

    def _on_view_direction_change(self, force=False):
        # all is flipped from zyx to xyz for vispy
        view_direction = np.sign(self.viewer.camera.view_direction)[::-1]
        up_direction = np.sign(self.viewer.camera.up_direction)[::-1]
        orientation_flip = [
            1 if ori == default_ori else -1
            for ori, default_ori in zip(
                self.viewer.camera.orientation,
                DEFAULT_ORIENTATION_TYPED,
                strict=True,
            )
        ][::-1]

        displayed = self.viewer.dims.displayed[::-1]
        ranges = [self.viewer.dims.range[i] for i in displayed]

        self.node.set_view_direction(
            ranges,
            list(view_direction),
            list(up_direction),
            orientation_flip,
            force=force,
        )

    def reset(self):
        super().reset()
        self._on_data_change()
