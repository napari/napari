from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.axes import AXES_COLORS
from napari._vispy.visuals.crosshair import Crosshair
from napari.utils.events import disconnect_events

if TYPE_CHECKING:
    from napari.components.overlays import CrosshairOverlay


class VispyCrosshairOverlay(ViewerOverlayMixin, VispySceneOverlay):
    """Overlay indicating the position of the cursor in the world."""

    overlay: CrosshairOverlay
    node: Crosshair

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            node=Crosshair(),
            **kwargs,
        )
        self.overlay.events.color.connect(self._on_color_change)
        self.overlay.events.colored.connect(self._on_color_change)
        self.viewer.dims.events.order.connect(self._on_color_change)
        self.viewer.dims.events.ndisplay.connect(self._on_color_change)
        self.overlay.events.gap.connect(self._on_gap_change)

        self.viewer.cursor.events.position.connect(self._on_position_change)
        self.viewer.dims.events.ndisplay.connect(self._on_position_change)
        self.viewer.dims.events.order.connect(self._on_position_change)
        self.viewer.dims.events.ndim.connect(self._on_position_change)

        self.reset()

    def _on_position_change(self) -> None:
        position = np.asarray(self.viewer.cursor.position, dtype=float)
        if position.size != self.viewer.dims.ndim:
            # cursor position is only updated on mouse move, so it can
            # temporarily be out of sync with the dims
            return
        displayed = list(self.viewer.dims.displayed[::-1])
        position = position[displayed]
        if len(displayed) == 2:
            position = np.append(position, 0)
        self.node.position = position

    def _on_color_change(self) -> None:
        if not self.overlay.colored:
            self.node.color = self.overlay.color
            return
        # mapped like axes overlays: the color depends on the
        # position of each displayed axis, counted from the end
        ndim = self.viewer.dims.ndim
        displayed = self.viewer.dims.displayed[::-1]
        colors = np.array(
            [AXES_COLORS[(ndim - 1 - a) % len(AXES_COLORS)] for a in displayed]
        )
        if len(colors) == 2:
            # z axis is unused in 2D
            colors = np.vstack([colors, np.asarray(self.overlay.color)])
        self.node.color = colors

    def _on_gap_change(self) -> None:
        self.node.gap = self.overlay.gap

    def close(self) -> None:
        disconnect_events(self.viewer.cursor.events, self)
        disconnect_events(self.viewer.dims.events, self)
        super().close()

    def reset(self) -> None:
        super().reset()
        self._on_position_change()
        self._on_color_change()
        self._on_gap_change()
