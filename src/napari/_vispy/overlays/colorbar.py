from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from napari._vispy.overlays.base import LayerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.colorbar import ColorBar
from napari.layers.utils.color_manager import ColorManager
from napari.settings import get_settings
from napari.utils.colormaps.colormap_utils import (
    _coerce_contrast_limits,
    _napari_cmap_to_vispy,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from napari._vispy.canvas import CanvasInfo
    from napari.components.overlays import ColorBarOverlay, Overlay
    from napari.layers import Image, Layer, Surface
    from napari.utils.colormaps import Colormap


class IntensityLayerWrapper:
    def __init__(self, layer: Image | Surface):
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
    def __init__(self, color_manager: ColorManager):
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
        layer: Image | Surface,
        canvas_info: CanvasInfo,
        **kwargs: Overlay,
    ) -> None:
        super().__init__(
            node=ColorBar(canvas_info=canvas_info),
            layer=layer,
            canvas_info=canvas_info,
            **kwargs,
        )
        self.layer: Layer
        self.x_size = 50
        self.y_size = 250

        self.source_wrapper: IntensityLayerWrapper | ColorManagerWrapper
        if self.overlay.colormanager_attribute is not None:
            color_manager = getattr(
                self.layer, self.overlay.colormanager_attribute
            )
            self.source_wrapper = ColorManagerWrapper(color_manager)
            color_manager.events.contrast_limits.connect(self._on_data_change)
            color_manager.events.continuous_colormap.connect(
                self._on_colormap_change
            )
        else:
            self.source_wrapper = IntensityLayerWrapper(self.layer)

            self.layer.events.colormap.connect(self._on_colormap_change)
            self.layer.events.contrast_limits.connect(self._on_data_change)
            self.layer.events.gamma.connect(self._on_gamma_change)

        self.overlay.events.size.connect(self._on_size_change)
        self.overlay.events.tick_length.connect(self._on_ticks_change)
        self.overlay.events.font_size.connect(self._on_ticks_change)
        self.overlay.events.box.connect(self._on_ticks_change)
        self.overlay.events.box_color.connect(self._on_ticks_change)
        self.overlay.events.color.connect(self._on_ticks_change)

        get_settings().appearance.events.theme.connect(self._on_data_change)
        self.viewer.events.theme.connect(self._on_data_change)

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
                clim=_coerce_contrast_limits(clim).contrast_limits,
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

        if self.overlay.color is not None:
            color = self.overlay.color
        else:
            color = self._get_fgcolor()

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
        self.y_size = self.overlay.size[1] + text_height / 2

        self._on_position_change()

    def reset(self) -> None:
        super().reset()
        self._on_data_change()
        self._on_colormap_change()
        self._on_size_change()
