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
        # one color per line, in napari order (z, y, x)
        color = self.overlay.color
        if color is None:
            # same as the axes overlay: colors are counted from the last
            # axis, and cycle when there are more axes than colors
            displayed = self.viewer.dims.displayed
            ndim = self.viewer.dims.ndim
            n = len(AXES_COLORS)
            colors = [AXES_COLORS[(ndim - 1 - axis) % n] for axis in displayed]
            if len(colors) == 2:
                # no z dim: the z line is only seen for 2D data in a 3D
                # camera (the shader drops it in a 2D camera), so color it
                # as the third axis would be
                colors.insert(0, AXES_COLORS[2])
        elif isinstance(color, tuple):
            colors = list(color)
        else:
            colors = [color] * 3
        # vispy order is x, y, z, i.e. the reverse of napari order
        self.node.color = np.array(colors[::-1])

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
