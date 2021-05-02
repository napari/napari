from vispy.scene.visuals import Compound

from .vispy_base_layer import VispyBaseLayer


class VispyLayerGroup(VispyBaseLayer):
    def __init__(self, layer):
        node = Compound([])
        super().__init__(layer, node)

    def _on_data_change(self, event=None):
        pass
