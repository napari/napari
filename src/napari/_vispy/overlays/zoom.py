"""Vispy zoom box overlay."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.interaction_box import InteractionBox
from napari.settings import get_settings

if TYPE_CHECKING:
    from napari.components.overlays import ZoomOverlay
    from napari.utils.events import Event


class VispyZoomOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    """Zoom box overlay.."""

    overlay: ZoomOverlay
    node: InteractionBox

    def __init__(self, **kwargs: Any):
        super().__init__(node=InteractionBox(), **kwargs)

        self.overlay.events.position.connect(self._on_position_change)

        self.reset()

    def _on_position_change(self, event: Event | None = None) -> None:
        """Change position."""
        settings = get_settings()
        self.node._highlight_width = (
            settings.appearance.highlight.highlight_thickness
        )
        self.node._edge_color = tuple(
            settings.appearance.highlight.highlight_color
        )  # type: ignore[assignment]

        top_left, bot_right = self.overlay.position
        self.node.set_data(
            # invert axes for vispy
            top_left[::-1],
            bot_right[::-1],
            handles=False,
            selected=None,
        )

    def reset(self) -> None:
        """Reset the overlay."""
        super().reset()
