from vispy.visuals.transforms import STTransform


class VispyBaseOverlay:
    def __init__(self, overlay, viewer, parent=None, node=None, order=0):
        super().__init__()
        self.viewer = viewer
        self.overlay = overlay
        self.node = node
        self.node.parent = parent  # qt_viewer.view
        self.node.order = order
        self.node.transform = STTransform()

        self.viewer.overlays.events.visible.connect(self._on_visible_change)
        self.overlay.events.visible.connect(self._on_visible_change)
        self.node.parent.events.resize.connect(self._on_position_change)

    def _on_visible_change(self):
        self.node.visible = (
            self.viewer.overlays.visible and self.overlay.visible
        )

    def _on_position_change(self, event=None):
        pass

    def reset(self):
        self._on_visible_change()
        self._on_position_change()
