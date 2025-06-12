"""Vispy zoom box overlay."""

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.interaction_box import InteractionBox
from napari.components.overlays.zoom import ZoomOverlay
from napari.settings import get_settings


class VispyZoomOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    """Zoom box overlay.."""

    def __init__(self, viewer, overlay: ZoomOverlay, parent=None):
        super().__init__(
            node=InteractionBox(),
            viewer=viewer,
            overlay=overlay,
            parent=parent,
        )

        self.overlay.events.canvas_positions.connect(self._on_positions_change)

        self._on_visible_change()

    def _on_positions_change(self, _evt=None):
        """Change position."""
        settings = get_settings()
        self.node._highlight_width = (
            settings.appearance.highlight.highlight_thickness
        )
        self.node._edge_color = settings.appearance.highlight.highlight_color

        # displayed = self.viewer.dims.displayed
        top_left, bot_right = self.overlay.canvas_positions
        self.node.set_data(
            # invert axes for vispy
            top_left[::-1],
            bot_right[::-1],
            handles=False,
            selected=None,
        )

    def reset(self):
        """Reset the overlay."""
        super().reset()
        self._on__change()
