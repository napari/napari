from vispy.color import Colormap as VispyColormap

from napari._vispy.overlays.base import VispyLayerCanvasOverlay
from napari._vispy.visuals.colormap import Colormap
from napari.utils.colormaps.colormap_utils import _coerce_contrast_limits


class VispyColormapOverlay(VispyLayerCanvasOverlay):
    def __init__(self, *, layer, overlay, parent=None) -> None:
        super().__init__(
            node=Colormap(), layer=layer, overlay=overlay, parent=parent
        )
        self.x_size = 50
        self.y_size = 250

        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.contrast_limits.connect(
            self._on_contrast_limits_change
        )
        self.layer.events.gamma.connect(self._on_gamma_change)

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

    def _on_colormap_change(self):
        self.node.set_cmap(VispyColormap(*self.layer.colormap))

    def _on_contrast_limits_change(self):
        self.node.set_clim(
            _coerce_contrast_limits(self.layer.contrast_limits).contrast_limits
        )

    def _on_gamma_change(self):
        self.node.set_gamma(self.layer.gamma)

    def reset(self):
        super().reset()
        self._on_data_change()
