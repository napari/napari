"""Vispy zoom box overlay."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.interaction_box import InteractionBox
from napari.settings import get_settings

if TYPE_CHECKING:
    from vispy.visuals.text.text import FontManager

    from napari.components.canvas import Canvas
    from napari.components.overlays import ZoomOverlay
    from napari.components.viewer_model import ViewerModel
    from napari.utils.events import Event


class VispyZoomOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    """Zoom box overlay.."""

    overlay: ZoomOverlay

    def __init__(
        self,
        viewer: ViewerModel,
        canvas: Canvas,
        overlay: ZoomOverlay,
        parent: Optional[Any] = None,
        font_manager: FontManager | None = None,
        font_family: str = 'OpenSans',
    ):
        super().__init__(
            node=InteractionBox(),
            viewer=viewer,
            canvas=canvas,
            overlay=overlay,
            parent=parent,
            font_manager=font_manager,
            font_family=font_family,
        )

        self.overlay.events.position.connect(self._on_position_change)

        self.reset()

    def _on_position_change(self, event: Optional[Event] = None) -> None:
        """Change position."""
        settings = get_settings()
        self.node._highlight_width = (
            settings.appearance.highlight.highlight_thickness
        )
        self.node._edge_color = settings.appearance.highlight.highlight_color

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
