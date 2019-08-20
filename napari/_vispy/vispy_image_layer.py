from vispy.scene.visuals import Image as ImageNode
from .vispy_base_layer import VispyBaseLayer


class VispyImageLayer(VispyBaseLayer):
    def __init__(self, layer):
        node = ImageNode(None, method='auto')
        super().__init__(layer, node)

        self.layer.events.interpolation.connect(
            lambda e: self._on_interpolation_change()
        )
        self.layer.events.colormap.connect(
            lambda e: self._on_colormap_change()
        )
        self.layer.events.clim.connect(lambda e: self._on_clim_change())

        self._on_interpolation_change()
        self._on_colormap_change()
        self._on_clim_change()
        self._on_data_change()

    def _on_data_change(self):
        self.node._need_colortransform_update = True
        self.node.set_data(self.layer._data_view)
        self.node.update()

    def _on_interpolation_change(self):
        self.node.interpolation = self.layer.interpolation

    def _on_colormap_change(self):
        self.node.cmap = self.layer.colormap[1]

    def _on_clim_change(self):
        self.node.clim = self.layer.clim
