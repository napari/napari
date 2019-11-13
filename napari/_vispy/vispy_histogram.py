from abc import ABC, abstractmethod

import numpy as np
from vispy.scene.visuals import Mesh as MeshNode

from ..components.histogram import Histogram as HistogramModel


"""Possible Organization

there is an analysis model (like HistogramModel here) that has a set_data
method (or setter).  The VispyPlotLayer is (currently) responsible both for
feeding the analysis model data from some layer and for creating and converting
the output of the model to some vispy Node that will be plotted in the view
widget of the NapariPlotWidget.
"""


class VispyPlotLayer(ABC):
    def __init__(self, layer, node):
        super().__init__()
        self.layer = layer
        self.node = node

    @abstractmethod
    def link_layer(self):
        raise NotImplementedError()


class VispyHistogramLayer(VispyPlotLayer):
    """

    link: can be data or view
    """

    def __init__(
        self,
        layer=None,
        link='data',
        bins=None,
        color=(1, 1, 1, 0.5),
        orientation='h',
    ):
        node = MeshNode()
        super().__init__(layer, node)

        self.color = color
        if link not in ('data', 'view'):
            raise ValueError('link must be either "data" or "view"')
        self.link = link

        if not isinstance(orientation, str) or orientation not in ('h', 'v'):
            raise ValueError(
                'orientation must be "h" or "v", not %s' % (orientation,)
            )
        self.orientation = orientation
        self.link_layer(layer)
        self.model = HistogramModel(bins=bins)
        self.model.events.data.connect(self._update_node)

    def update_model(self, *args):
        if self.link == 'data':
            self.model.set_data(self.layer.data)
        elif self.link == 'view':
            self.model.set_data(self.layer._data_raw)

    def link_layer(self, layer):
        self.layer = layer
        if layer is None:
            return
        self.layer.events.set_data.disconnect(self.update_model)
        self.layer.events.data.disconnect(self.update_model)

        if self.link == 'data':
            self.layer.events.data.connect(self.update_model)
        elif self.link == 'view':
            self.layer.events.set_data.connect(self.update_model)
        self.update_model()

    def _update_node(self, event):
        counts, edges = event.counts, event.edges
        X, Y = (0, 1) if self.orientation == 'h' else (1, 0)
        # construct our vertices
        verts = np.zeros((3 * len(edges) - 2, 3), np.float32)
        verts[:, X] = np.repeat(edges, 3)[1:-1]
        verts[1::3, Y] = counts
        verts[2::3, Y] = counts
        edges.astype(np.float32)
        # and now our tris
        faces = np.zeros((2 * len(edges) - 2, 3), np.uint32)
        offsets = 3 * np.arange(len(edges) - 1, dtype=np.uint32)[:, np.newaxis]
        tri_1 = np.array([0, 2, 1])
        tri_2 = np.array([2, 0, 3])
        faces[::2] = tri_1 + offsets
        faces[1::2] = tri_2 + offsets

        vert_colors = np.tile(np.array([1, 1, 1, 0.6]), (len(verts), 1))
        vert_colors[verts[:, Y] == 0] = np.array([0.18, 0.10, 0.22, 0.2])
        self.node.set_data(verts, faces, vertex_colors=vert_colors)
        # self.node.update()
        return verts, faces
