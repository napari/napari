from typing import Type

import numpy as np
import pytest
from napari_graph import BaseGraph, DirectedGraph, UndirectedGraph

from napari._vispy.layers.graph import VispyGraphLayer
from napari.layers import Graph


@pytest.mark.parametrize("graph_class", [UndirectedGraph, DirectedGraph])
def test_vispy_graph_layer(graph_class: Type[BaseGraph]) -> None:
    edges = np.asarray([[0, 1], [1, 2]])
    coords = np.asarray([[0, 0, 0, -1], [0, 0, 1, 2], [1, 0, 2, 3]])

    graph = graph_class(edges=edges, coords=coords)

    layer = Graph(graph)
    visual = VispyGraphLayer(layer)

    # checking nodes positions
    assert np.all(
        coords[:2, 1:]
        == np.flip(visual.node._subvisuals[0]._data["a_position"], axis=-1)
    )

    # checking edges positions
    assert np.all(
        coords[:2, 2:] == np.flip(visual.node._subvisuals[4]._pos, axis=-1)
    )
