import numpy as np

from ..visuals.vectors import VectorsVisual
from .base import VispyBaseLayer


class VispyVectorsLayer(VispyBaseLayer):
    def __init__(self, layer):
        node = VectorsVisual(connect='segments', antialias=False)
        super().__init__(layer, node)

        self.layer.events.edge_color.connect(self._on_data_change)
        self.layer.events.length.connect(self._on_data_change)
        self.layer.events.edge_width.connect(self._on_edge_width_change)

        self.reset()
        self._on_data_change()

    def _on_data_change(self, event=None):
        if len(self.layer._view_data) == 0:
            pos = np.zeros((1, 2, self.layer._ndisplay))
            color = np.zeros((0, 4))
        else:
            # reverse to draw most recent last and swap xy for vispy
            pos = self.layer._view_data[::-1, :, ::-1].copy()
            # scale vector and add it to its origin to get the endpoint coordinate
            pos[:, 1] *= self.layer.length
            pos[:, 1] += pos[:, 0]
            color = self.layer._view_color

        if self.layer._ndisplay == 3 and self.layer.ndim == 2:
            pos = np.pad(pos, ((0, 0), (0, 0), (0, 1)), mode='constant')

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

    def _on_edge_width_change(self, event=None):
        self.node.set_data(width=self.layer.edge_width)

    def reset(self):
        self._reset_base()
        self._on_data_change()
