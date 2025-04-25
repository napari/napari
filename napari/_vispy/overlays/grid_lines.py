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

        self.overlay.events.color.connect(self._on_data_change)
        self.overlay.events.tick_labels.connect(self._on_data_change)
        self.overlay.events.tick_spacing.connect(self._on_data_change)
        self.viewer.dims.events.order.connect(self._on_data_change)
        self.viewer.dims.events.range.connect(self._on_data_change)
        self.viewer.dims.events.ndisplay.connect(self._on_data_change)

        # would be nice to fire this less often to save performance
        self.viewer.camera.events.angles.connect(
            self._on_view_direction_change
        )

        self.reset()

    def _on_data_change(self):
        # napari dims are zyx, but vispy uses xyz
        displayed = self.viewer.dims.displayed[::-1]
        ranges = [self.viewer.dims.range[i] for i in displayed]

        if self.overlay.tick_spacing == 'auto':
            spacing = 'auto'
        else:
            spacing = [self.overlays.tick_spacing[i] for i in displayed]

        self.node.reset_grids(self.overlay.color)
        self.node.set_extents(ranges)
        self.node.set_ticks(self.overlay.tick_labels, spacing, ranges)

        self._on_view_direction_change()

    def _on_view_direction_change(self):
        # flip to xyz for vispy
        view_direction = np.array(self.viewer.camera.view_direction)[::-1] >= 0
        up_direction = np.sign(self.viewer.camera.up_direction)[::-1]

        displayed = self.viewer.dims.displayed[::-1]
        ranges = [self.viewer.dims.range[i] for i in displayed]

        self.node.set_view_direction(
            ranges,
            list(view_direction),
            list(up_direction),
        )

    def reset(self):
        super().reset()
        self._on_data_change()
