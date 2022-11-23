from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.interaction_box import InteractionBox


class VispyInteractionBoxOverlay(LayerOverlayMixin, VispySceneOverlay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, node=InteractionBox(), **kwargs)
        self.overlay.events.bounds.connect(self._on_bounds_change)
        self.overlay.events.handles.connect(self._on_bounds_change)
        self.overlay.events.selected_vertex.connect(self._on_bounds_change)

    def _on_bounds_change(self):
        top_left, bot_right = self.overlay.bounds
        self.node.set_data(
            # invert axes for vispy
            top_left[::-1],
            bot_right[::-1],
            handles=self.overlay.handles,
            selected=self.overlay.selected_vertex,
        )
