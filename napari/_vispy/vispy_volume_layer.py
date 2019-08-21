import numpy as np
from vispy.scene.visuals import Volume as VolumeNode

from ..layers import Volume
from .vispy_base_layer import VispyBaseLayer


class VispyVolumeLayer(VispyBaseLayer, layer=Volume):
    def __init__(self, layer):
        node = VolumeNode(np.empty((1, 1, 1)))
        super().__init__(layer, node)

        self.layer.events.rendering.connect(
            lambda e: self._on_rendering_change()
        )
        self.layer.events.colormap.connect(
            lambda e: self._on_colormap_change()
        )
        self.layer.events.clim.connect(lambda e: self._on_data_change())

        self._on_rendering_change()
        self._on_colormap_change()
        self._on_data_change()

    def _on_data_change(self):
        self.node.set_data(self.layer._data_view, clim=self.layer.clim)
        self.node.update()

    def _on_rendering_change(self):
        self.node.method = self.layer.rendering

    def _on_colormap_change(self):
        cmap = self.layer.colormap[1]
        self.node.view_program['texture2D_LUT'] = (
            cmap.texture_lut() if (hasattr(cmap, 'texture_lut')) else None
        )
        self.node.cmap = cmap
