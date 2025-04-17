from vispy.color import Colormap as VispyColormap

from napari._vispy.overlays.base import LayerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.colormap import Colormap
from napari.utils.colormaps.colormap_utils import _coerce_contrast_limits


class VispyColormapOverlay(LayerOverlayMixin, VispyCanvasOverlay):
    def __init__(self, *, layer, overlay, parent=None) -> None:
        super().__init__(
            node=Colormap(), layer=layer, overlay=overlay, parent=parent
        )
        self.x_size = 50
        self.y_size = 250
        self.x_offset = 7
        self.y_offset = 7

        self.layer.events.contrast_limits.connect(self._on_data_change)
        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.gamma.connect(self._on_gamma_change)
        self.overlay.events.size.connect(self._on_size_change)
        self.overlay.events.ticks.connect(self._on_ticks_change)
        self.overlay.events.n_ticks.connect(self._on_ticks_change)
        self.overlay.events.tick_length.connect(self._on_ticks_change)
        self.overlay.events.font_size.connect(self._on_ticks_change)
        self.overlay.events.color.connect(self._on_ticks_change)

        self.reset()

    def _on_data_change(self):
        self.node.set_data_and_clim(
            clim=_coerce_contrast_limits(
                self.layer.contrast_limits
            ).contrast_limits,
            dtype=self.layer.data.dtype,
        )
        self._on_colormap_change()
        self._on_gamma_change()
        self._on_ticks_change()

    def _on_colormap_change(self):
        self.node.set_cmap(VispyColormap(*self.layer.colormap))

    def _on_gamma_change(self):
        self.node.set_gamma(self.layer.gamma)

    def _on_size_change(self):
        self.node.set_size(self.overlay.size)
        self._on_ticks_change()

    def _on_ticks_change(self):
        text_width = self.node.set_ticks(
            show=self.overlay.ticks,
            n=self.overlay.n_ticks,
            tick_length=self.overlay.tick_length,
            size=self.overlay.size,
            font_size=self.overlay.font_size,
            clim=_coerce_contrast_limits(
                self.layer.contrast_limits
            ).contrast_limits,
            color=self.overlay.color,
        )

        if self.overlay.ticks:
            # 7 is the base, 0.8 is just a magic number to scale font size
            self.x_size = (
                self.overlay.size[0]
                + self.overlay.tick_length
                + text_width * 0.8
            )
            self.y_offset = 7 + self.overlay.font_size
        else:
            self.x_size = self.overlay.size[0]
            self.y_offset = 7

        self._on_position_change()

    def reset(self):
        super().reset()
        self._on_data_change()
        self._on_size_change()
