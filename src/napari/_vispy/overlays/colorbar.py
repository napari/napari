from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from napari._vispy.overlays.base import LayerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.colorbar import Colorbar
from napari.layers.utils.color_manager import ColorManager
from napari.settings import get_settings
from napari.utils.colormaps.colormap_utils import (
    _coerce_contrast_limits,
    _napari_cmap_to_vispy,
)
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from vispy.scene import Node

    from napari.components.overlays import ColorBarOverlay, Overlay
    from napari.layers import Layer
    from napari.utils.colormaps import Colormap


class IntensityLayerWrapper:
    def __init__(self, overlay, layer: Layer):
        self.layer = layer

    @property
    def contrast_limits(self) -> tuple[float, float]:
        return self.layer.contrast_limits

    @property
    def dtype(self) -> DTypeLike:
        return self.layer.dtype

    @property
    def gamma(self) -> float:
        return self.layer.gamma

    @property
    def colormap(self) -> Colormap:
        return self.layer.colormap


class ColorManagerWrapper:
    def __init__(self, overlay, color_manager: ColorManager):
        self.color_manager = color_manager

    @property
    def contrast_limits(self) -> tuple[float, float] | None:
        return self.color_manager.contrast_limits

    @property
    def dtype(self) -> DTypeLike:
        return np.float32

    @property
    def gamma(self) -> float:
        return 1

    @property
    def colormap(self) -> Colormap:
        # categorical colormap not yet supported
        return self.color_manager.continuous_colormap


class VispyColorBarOverlay(LayerOverlayMixin, VispyCanvasOverlay):
    overlay: ColorBarOverlay

    def __init__(
        self,
        *,
        layer: Layer,
        overlay: Overlay,
        parent: Node | None = None,
    ) -> None:
        super().__init__(
            node=Colorbar(), layer=layer, overlay=overlay, parent=parent
        )
        self.layer: Layer
        self.x_size = 50
        self.y_size = 250
        self.x_offset = 7
        self.y_offset = 7

        if self.overlay.colormanager_attribute is not None:
            color_manager = getattr(
                self.layer, self.overlay.colormanager_attribute
            )
            self.source_wrapper = ColorManagerWrapper(
                self.overlay, color_manager
            )
            color_manager.events.contrast_limits.connect(
                self._on_data_change
            )  # will it work to not have face_contrast_limits as that is what the clims are called??
            # TODO: connect other colormanager events
            color_manager.events.continuous_colormap.connect(
                self._on_colormap_change
            )
            # color_manager.events.gamma.connect(self._on_gamma_change)
        else:
            self.source_wrapper = IntensityLayerWrapper(
                self.overlay, self.layer
            )

            self.layer.events.colormap.connect(self._on_colormap_change)
            self.layer.events.contrast_limits.connect(self._on_data_change)
            self.layer.events.gamma.connect(self._on_gamma_change)

        self.overlay.events.size.connect(self._on_size_change)
        self.overlay.events.tick_length.connect(self._on_ticks_change)
        self.overlay.events.font_size.connect(self._on_ticks_change)
        self.overlay.events.color.connect(self._on_ticks_change)

        get_settings().appearance.events.theme.connect(self._on_data_change)

        self.reset()

        self._on_data_change()

    def _on_visible_change(self) -> None:
        super()._on_visible_change()
        # necessary to update outdated values since we skip updating when
        # invisible
        self._on_gamma_change()
        self._on_ticks_change()

    def _on_data_change(self) -> None:
        # TODO: this branching is unfortunately necessary for now until we
        #       support some kind of colorbar for categorical data
        # currently unsupported path of categorical colormap
        # we just make invisible and bail out, and everywhere else
        # we make sure to not update things when invisible
        clim = self.source_wrapper.contrast_limits
        if clim is None:
            self.node.visible = False
        else:
            self._on_visible_change()
            self.node.set_data_and_clim(
                clim=_coerce_contrast_limits(
                    self.source_wrapper.contrast_limits
                ).contrast_limits,
                dtype=self.source_wrapper.dtype,
            )

    def _on_gamma_change(self) -> None:
        if self.node.visible:
            self.node.set_gamma(self.source_wrapper.gamma)

    def _on_colormap_change(self) -> None:
        colormap = self.source_wrapper.colormap
        self.node.set_cmap(_napari_cmap_to_vispy(colormap))

    def _on_size_change(self) -> None:
        self.node.set_size(self.overlay.size)
        self._on_ticks_change()

    def _on_ticks_change(self) -> None:
        # set color to the negative of theme background.
        # the reason for using the `as_hex` here is to avoid
        # `UserWarning` which is emitted when RGB values are above 1
        if self.source_wrapper.contrast_limits is None:
            return

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
                self.source_wrapper.contrast_limits
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
        self._on_colormap_change()
        self._on_size_change()
