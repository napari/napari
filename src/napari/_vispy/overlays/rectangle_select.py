"""Vispy rectangle select overlay."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.interaction_box import InteractionBox
from napari.settings import get_settings

if TYPE_CHECKING:
    from napari.components.overlays import RectangleSelectOverlay
    from napari.utils.events import Event


class VispyRectangleSelectOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    """A selection rectangle that lives in canvas space."""

    overlay: RectangleSelectOverlay
    node: InteractionBox

    def __init__(self, **kwargs: Any):
        super().__init__(node=InteractionBox(), **kwargs)

        self.overlay.events.corners_canvas.connect(self._on_corners_change)

        self.reset()

    def _on_corners_change(self, event: Optional[Event] = None) -> None:
        """Change position."""
        settings = get_settings()
        self.node.highlight_width = (
            settings.appearance.highlight.highlight_thickness
        )
        self.node.highlight_color = tuple(  # type: ignore
            settings.appearance.highlight.highlight_color
        )

        top_left, bot_right = self.overlay.corners_canvas
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
