import numpy as np

from ..visuals.vectors import VectorsVisual
from .base import VispyBaseLayer


class VispyVectorsLayer(VispyBaseLayer):
    def __init__(self, layer):
        node = VectorsVisual()
        super().__init__(layer, node)

        self.layer.events.edge_color.connect(self._on_data_change)

        self.reset()
        self._on_data_change()

    def _on_data_change(self, event=None):
        if len(self.layer._view_data) == 0:
            pos = np.empty((0, 2, self.layer._ndisplay))
            color = np.empty((0, 4))
        else:
            # reverse to draw most recent last
            pos = self.layer._view_data[::-1].copy()
            # add vector to origin for second point
            pos[:, 1] += pos[:, 0]
            color = self.layer._view_color

        if self.layer._ndisplay == 3 and self.layer.ndim == 2:
            pos = np.pad(pos, ((0, 0), (0, 1)), mode='constant')

        # reshape to what LineVisual needs
        pos = pos.reshape(-1, self.layer._ndisplay)
        color = color.repeat(2, axis=0)

        self.node.set_data(
            pos=pos,
            color=color,
        )

        self.node.update()
        # Call to update order of translation values with new dims:
        self._on_matrix_change()
