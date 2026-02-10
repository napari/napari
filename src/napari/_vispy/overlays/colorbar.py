from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from napari._vispy.overlays.base import LayerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.colormap import Colormap
from napari.layers.utils.color_manager import ColorManager
from napari.settings import get_settings
from napari.utils.colormaps.colormap_utils import (
    _coerce_contrast_limits,
    _napari_cmap_to_vispy,
)
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from vispy.scene import Node

    from napari.components.overlays import ColorBarOverlay, Overlay
    from napari.layers import Image, Points, Surface


class VispyColorBarOverlay(LayerOverlayMixin, VispyCanvasOverlay):
    overlay: ColorBarOverlay

    def __init__(
        self,
        *,
        layer: Image | Surface | Points,
        overlay: Overlay,
        color_manager: ColorManager,
        parent: Node | None = None,
    ) -> None:
        super().__init__(
            node=Colormap(), layer=layer, overlay=overlay, parent=parent
        )
        self.layer: Image | Surface | Points
        self.color_manager = (
            color_manager if color_manager is not None else self.layer
        )
        self.x_size = 50
        self.y_size = 250
        self.x_offset = 7
        self.y_offset = 7
        # TODO: check with napari core whether image and surface layer always have contrast limits that are not None.
        # Checking with the points layer, this layer can have face_contrast_limits set to None.
        if getattr(self.layer, 'contrast_limits', None):
            self.color_manager.events.contrast_limits.connect(
                self._on_data_change
            )
            self.layer.events.colormap.connect(self._on_colormap_change)
            self.layer.events.gamma.connect(self._on_gamma_change)
        else:
            self.layer.events.face_contrast_limits.connect(
                self._on_data_change
            )
            self.layer.events.face_colormap.connect(self._on_colormap_change)

        self.overlay.events.visible.connect(
            self._check_contrast_limits_colorbar
        )
        self.overlay.events.size.connect(self._on_size_change)
        self.overlay.events.tick_length.connect(self._on_ticks_change)
        self.overlay.events.font_size.connect(self._on_ticks_change)
        self.overlay.events.color.connect(self._on_ticks_change)

        get_settings().appearance.events.theme.connect(self._on_data_change)

        self.reset()

    def _check_contrast_limits_colorbar(self) -> None:
        if self.color_manager.contrast_limits is None:
            warnings.warn(
                'Colorbar overlay is set to visible but the layer has no '
                'contrast limits set. Hiding colorbar overlay.',
                UserWarning,
            )
            self.layer.colorbar.visible = False

    def _on_data_change(self) -> None:
        if self.color_manager.contrast_limits is not None:
            # TODO: for initial implementation only focus on face_color of points layer and not border.
            dtype = (
                getattr(self.layer, 'dtype', None)
                or self.layer.face_color.dtype
            )
            self.node.set_data_and_clim(
                clim=_coerce_contrast_limits(
                    self.color_manager.contrast_limits
                ).contrast_limits,
                dtype=dtype,
            )
            self._on_colormap_change()
            if getattr(self.layer, 'contrast_limits', None):
                self._on_gamma_change()
            self._on_ticks_change()
        else:
            warnings.warn(
                'Colorbar overlay is set to visible but the layer has no '
                'contrast limits set. Hiding colorbar overlay.',
                UserWarning,
            )
            self.layer.colorbar.visible = False

    def _on_colormap_change(self) -> None:
        colormap = (
            getattr(self.layer, 'colormap', None) or self.layer.face_colormap
        )
        self.node.set_cmap(_napari_cmap_to_vispy(colormap))

    def _on_gamma_change(self) -> None:
        self.node.set_gamma(self.layer.gamma)

    def _on_size_change(self) -> None:
        self.node.set_size(self.overlay.size)
        self._on_ticks_change()

    def _on_ticks_change(self) -> None:
        # set color to the negative of theme background.
        # the reason for using the `as_hex` here is to avoid
        # `UserWarning` which is emitted when RGB values are above 1
        if (
            getattr(self.layer, 'contrast_limits', None)
            or self.layer.face_contrast_limits
        ):
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
                    getattr(self.layer, 'contrast_limits', None)
                    or self.layer.face_contrast_limits
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
