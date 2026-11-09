import numpy as np

from networkx import Graph
from napari.layers.base import Layer

from napari.layers._graph._slice import _GraphSlicingState


class _Graph(Layer):
    _data: Graph

    def __init__(self, graph=None, ndim: int | None = None):
        inferred_ndim: int | None = self._get_graph_ndim(graph)

        if (
            ndim is not None
            and inferred_ndim is not None
            and inferred_ndim != ndim
        ):
            raise ValueError(
                f'The graph nodes dimensions {inferred_ndim} do not match the provided ndim argument {ndim}'
            )

        ndim = (
            inferred_ndim
            if inferred_ndim is not None
            else (2 if ndim is None else ndim)
        )

        self._data = graph
        super().__init__(data=graph, ndim=ndim)
        self.set_view_slice()

    @property
    def data(self) -> Graph:
        return self._data

    @data.setter
    def data(self, data: Graph) -> None:
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

    @staticmethod
    def _get_graph_ndim(graph: Graph) -> int | None:
        nodes = list(graph.nodes(data=True))

        if nodes:
            if not all('coordinates' in node_attr for _, node_attr in nodes):
                raise ValueError('All nodes need to have coordinates')
            if len({len(node[1]['coordinates']) for node in nodes}) != 1:
                raise ValueError(
                    'All node coordinates need to have the same ndim'
                )

        return len(nodes[0][1]['coordinates']) if nodes else None

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
