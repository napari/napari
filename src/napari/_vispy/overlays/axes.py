from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from napari import Viewer
from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.axes import Axes
from napari.components.overlays import AxesOverlay
from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from vispy.scene import ViewBox
    from vispy.visuals.text.text import FontManager


class VispyAxesOverlay(ViewerOverlayMixin, VispySceneOverlay):
    """Axes indicating world coordinate origin and orientation."""

    overlay: AxesOverlay

    def __init__(
        self,
        *,
        viewer: Viewer,
        overlay: AxesOverlay,
        parent: ViewBox | None = None,
        font_manager: FontManager | None = None,
        font_family: str = 'OpenSans',
    ) -> None:
        self._scale = 1.0

        # Target axes length in canvas pixels
        self._target_length = 80

        super().__init__(
            node=Axes(font_manager=font_manager, font_family=font_family),
            viewer=viewer,
            overlay=overlay,
            parent=parent,
            font_manager=font_manager,
            font_family=font_family,
        )
        self.overlay.events.colored.connect(self._on_data_change)
        self.overlay.events.dashed.connect(self._on_data_change)
        self.overlay.events.labels.connect(self._on_labels_visible_change)
        self.overlay.events.arrows.connect(self._on_data_change)

        self.viewer.events.theme.connect(self._on_data_change)
        self.viewer.camera.events.zoom.connect(self._on_zoom_change)
        self.viewer.dims.events.order.connect(self._on_data_change)
        self.viewer.dims.events.range.connect(self._on_data_change)
        self.viewer.dims.events.ndisplay.connect(self._on_data_change)
        self.viewer.dims.events.axis_labels.connect(
            self._on_labels_text_change
        )

        self.reset()

    def _on_data_change(self) -> None:
        # Determine which axes are displayed
        axes = self.viewer.dims.displayed[::-1]

        # Counting backwards from total number of dimensions
        # determine axes positions. This is done as by default
        # the last NumPy axis corresponds to the first Vispy axis
        reversed_axes = [self.viewer.dims.ndim - 1 - a for a in axes]

        self.node.set_data(
            axes=axes,
            reversed_axes=reversed_axes,
            colored=self.overlay.colored,
            bg_color=get_theme(self.viewer.theme).canvas,
            dashed=self.overlay.dashed,
            arrows=self.overlay.arrows,
        )

        self._on_labels_text_change()

    def _on_labels_visible_change(self) -> None:
        self.node.text.visible = self.overlay.labels

    def _on_labels_text_change(self) -> None:
        axes = self.viewer.dims.displayed[::-1]
        axis_labels = [self.viewer.dims.axis_labels[a] for a in axes]
        self.node.text.text = axis_labels

    def _on_zoom_change(self) -> None:
        scale = 1 / self.viewer.camera.zoom

        # If scale has not changed, do not redraw
        if abs(np.log10(self._scale) - np.log10(scale)) < 1e-4:
            return
        self._scale = scale
        scale = self._target_length * self._scale
        # Update axes scale
        self.node.transform.reset()
        self.node.transform.scale([scale, scale, scale, 1])

    def reset(self) -> None:
        super().reset()
        self._on_data_change()
        self._on_labels_visible_change()
        self._on_zoom_change()
