from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from napari._vispy.overlays.base import LayerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.colormap import Colormap
from napari.settings import get_settings
from napari.utils.colormaps.colormap_utils import (
    _coerce_contrast_limits,
    _napari_cmap_to_vispy,
)
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from vispy.scene import Node

    from napari.components.overlays import Overlay
    from napari.layers import Image, Surface


class VispyColorBarOverlay(LayerOverlayMixin, VispyCanvasOverlay):
    def __init__(
        self,
        *,
        layer: Image | Surface,
        overlay: Overlay,
        parent: Node | None = None,
    ) -> None:
        super().__init__(
            node=Colormap(), layer=layer, overlay=overlay, parent=parent
        )
        self.layer: Image | Surface
        self.x_size = 50
        self.y_size = 250
        self.x_offset = 7
        self.y_offset = 7

        self.layer.events.contrast_limits.connect(self._on_data_change)
        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.gamma.connect(self._on_gamma_change)
        self.overlay.events.size.connect(self._on_size_change)
        self.overlay.events.tick_length.connect(self._on_ticks_change)
        self.overlay.events.font_size.connect(self._on_ticks_change)
        self.overlay.events.color.connect(self._on_ticks_change)

        get_settings().appearance.events.theme.connect(self._on_data_change)

        self.reset()

    def _on_data_change(self) -> None:
        self.node.set_data_and_clim(
            clim=_coerce_contrast_limits(
                self.layer.contrast_limits
            ).contrast_limits,
            dtype=self.layer.dtype,
        )
        self._on_colormap_change()
        self._on_gamma_change()
        self._on_ticks_change()

    def _on_colormap_change(self) -> None:
        self.node.set_cmap(_napari_cmap_to_vispy(self.layer.colormap))

    def _on_gamma_change(self) -> None:
        self.node.set_gamma(self.layer.gamma)

    def _on_size_change(self) -> None:
        self.node.set_size(self.overlay.size)
        self._on_ticks_change()

    def _on_ticks_change(self) -> None:
        # set color to the negative of theme background.
        # the reason for using the `as_hex` here is to avoid
        # `UserWarning` which is emitted when RGB values are above 1
        color = self.overlay.color
        if color is None:
            if (
                self.node.parent is not None
                and self.node.parent.canvas.bgcolor
            ):
                background_color = self.node.parent.canvas.bgcolor.rgba
            else:
                background_color = get_theme(
                    get_settings().appearance.theme
                ).canvas.as_hex()
                background_color = transform_color(background_color)[0]
            color = np.subtract(1, background_color)
            color[-1] = background_color[-1]

        text_width, text_height = self.node.set_ticks_and_get_text_size(
            tick_length=self.overlay.tick_length,
            font_size=self.overlay.font_size,
            clim=_coerce_contrast_limits(
                self.layer.contrast_limits
            ).contrast_limits,
            color=color,
        )

        # Calculate proper layout with explicit spacing constants
        self.x_size = (
            self.overlay.size[0]  # Colorbar width
            + self.overlay.tick_length  # Tick marks length
            + text_width  # Text width with margins
        )
        self.y_size = text_height

        self._on_position_change()

    def reset(self) -> None:
        super().reset()
        self._on_data_change()
        self._on_size_change()
