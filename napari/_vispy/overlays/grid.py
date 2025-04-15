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

        self.viewer.dims.events.order.connect(self._on_data_change)
        self.viewer.dims.events.range.connect(self._on_data_change)
        self.viewer.dims.events.ndisplay.connect(self._on_data_change)

        self.reset()

    def _on_data_change(self):
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

    def reset(self):
        super().reset()
        self._on_data_change()
