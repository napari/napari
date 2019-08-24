from vispy.scene.visuals import Image as ImageNode

from ..layers import Labels
from .vispy_base_layer import VispyBaseLayer


class VispyLabelsLayer(VispyBaseLayer, layer=Labels):
    def __init__(self, layer):
        node = ImageNode(None, method='auto')
        super().__init__(layer, node)

        self.node.cmap = self.layer.colormap[1]
        self.node.clim = [0.0, 1.0]
        self.reset()

    def _on_data_change(self):
        self.node._need_colortransform_update = True
        image = self.layer._raw_to_displayed(self.layer._data_view)
        self.node.set_data(image)
        self.node.update()

    def reset(self):
        self._reset_base()
        self._on_data_change()
