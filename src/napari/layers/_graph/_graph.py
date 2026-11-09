import numpy as np
from networkx import Graph as nxGraph

from napari.layers._graph._graph_utils import fix_graph_data
from napari.layers._graph._slice import _GraphSlicingState
from napari.layers.base import Layer


class _Graph(Layer):
    _data: nxGraph

    def __init__(self, data=None, ndim: int | None = None):

        data, ndim = fix_graph_data(data, ndim)

        self._data = data
        super().__init__(data=data, ndim=ndim)
        self.set_view_slice()

    @property
    def data(self) -> nxGraph:
        return self._data

    @data.setter
    def data(self, data: nxGraph) -> None:
        self._data = data

    def _get_ndim(self) -> int:
        nodes = self.data.nodes(data=True)
        first_node = next(iter(nodes), None)

        return (
            self._ndim
            if first_node is None
            else len(first_node[1]['coordinates'])
        )

    @property
    def _extent_data(self) -> np.ndarray:
        if not self.data.nodes:
            return np.full((2, self.ndim), np.nan)
        coordinates = np.asarray(
            [
                node_attr['coordinates']
                for _, node_attr in self.data.nodes(data=True)
            ]
        )
        min_coords = np.min(coordinates, axis=0)
        max_coords = np.max(coordinates, axis=0)

        return np.vstack([min_coords, max_coords])

    def _get_layer_slicing_state(self, data, cache) -> _GraphSlicingState:
        return _GraphSlicingState(self, data, cache)

    def _get_state(self) -> dict:
        state = self._get_base_state()
        state.update({'data': self.data, 'ndim': self.ndim})
        return state

    def _set_view_slice(self) -> None:
        pass

    def _update_thumbnail(self) -> None:
        thumbnail = np.zeros(self._thumbnail_shape)
        thumbnail[..., 3] = self.opacity
        self.thumbnail = thumbnail

    def _get_value(self, position):
        return None
