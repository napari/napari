from vispy.scene.visuals import Mesh as MeshNode
from .vispy_base_layer import VispyBaseLayer
import numpy as np


class VispyVectorsLayer(VispyBaseLayer):
    def __init__(self, layer):
        node = MeshNode()
        super().__init__(layer, node)

        self.layer.events.edge_color.connect(lambda e: self._on_data_change())

        self.reset()

    def _on_data_change(self):
        if (
            len(self.layer._view_vertices) == 0
            or len(self.layer._view_faces) == 0
        ):
            vertices = np.zeros((3, self.layer.dims.ndisplay))
            faces = np.array([[0, 1, 2]])
        else:
            vertices = self.layer._view_vertices[:, ::-1] + 0.5
            faces = self.layer._view_faces

        if self.layer.dims.ndisplay == 3 and self.layer.dims.ndim == 2:
            vertices = np.pad(vertices, ((0, 0), (0, 1)))

        self.node.set_data(
            vertices=vertices, faces=faces, color=self.layer.edge_color
        )
        self.node.update()

    def reset(self):
        self._reset_base()
        self._on_data_change()
