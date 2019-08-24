from vispy.scene.visuals import Mesh as MeshNode

from ..layers import Vectors
from .vispy_base_layer import VispyBaseLayer


class VispyVectorsLayer(VispyBaseLayer, layer=Vectors):
    def __init__(self, layer):
        node = MeshNode()
        super().__init__(layer, node)

        self.layer.events.edge_color.connect(lambda e: self._on_data_change())

        self.reset()

    def _on_data_change(self):
        self.node.set_data(
            vertices=self.layer._view_vertices,
            faces=self.layer._view_faces,
            color=self.layer.edge_color,
        )
        self.node.update()

    def reset(self):
        self._reset_base()
        self._on_data_change()
