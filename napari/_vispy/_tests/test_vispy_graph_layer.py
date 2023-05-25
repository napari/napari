from typing import Type

import numpy as np
import pytest

from napari._vispy.layers.graph import VispyGraphLayer
from napari.layers import Graph

pytest.importorskip("napari_graph")

from napari_graph import (  # noqa: E402
    BaseGraph,
    DirectedGraph,
    UndirectedGraph,
)


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


@pytest.mark.parametrize("graph_class", [UndirectedGraph, DirectedGraph])
def test_vispy_graph_layer_removal(graph_class: Type[BaseGraph]) -> None:
    edges = np.asarray([[0, 1], [1, 2]])
    coords = np.asarray([[0, 0, 0, -1], [0, 0, 1, 2], [0, 0, 2, 3]])

    graph = graph_class(edges=edges, coords=coords)

    layer = Graph(graph)
    visual = VispyGraphLayer(layer)

    # checking nodes positions
    assert np.all(
        coords[:, 1:]
        == np.flip(visual.node._subvisuals[0]._data["a_position"], axis=-1)
    )

    # checking first edge
    assert np.all(
        coords[:2, 2:] == np.flip(visual.node._subvisuals[4]._pos[:2], axis=-1)
    )

    # checking second edge
    assert np.all(
        coords[1:3, 2:]
        == np.flip(visual.node._subvisuals[4]._pos[2:], axis=-1)
    )

    layer.remove(0)

    # checking remaining nodes positions
    assert np.all(
        coords[1:, 1:]
        == np.flip(visual.node._subvisuals[0]._data["a_position"], axis=-1)
    )

    # checking single edge
    assert np.all(
        coords[1:3, 2:] == np.flip(visual.node._subvisuals[4]._pos, axis=-1)
    )
