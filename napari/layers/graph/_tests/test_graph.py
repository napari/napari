from typing import Type
from unittest.mock import Mock

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from napari_graph import (
    BaseGraph,
    DirectedGraph,
    UndirectedGraph,
)

from napari.layers import Graph
from napari.layers.base._base_constants import ActionType


def test_empty_graph() -> None:
    graph = Graph()
    assert len(graph.data) == 0


def test_2_dim_array_graph() -> None:
    shape = (5, 2)

    np.random.seed(1)
    graph = Graph(np.random.random(shape))

    assert len(graph.data) == shape[0]
    assert graph.ndim == shape[1]


def test_3_dim_array_graph() -> None:
    shape = (5, 2, 2)

    np.random.seed(1)
    with pytest.raises(ValueError):
        Graph(np.random.random(shape))


def test_incompatible_data_graph() -> None:
    dict_graph = {0: [], 1: [1], 2: [0, 1]}

    with pytest.raises(TypeError):
        Graph(dict_graph)


def test_non_spatial_graph() -> None:
    non_spatial_graph = UndirectedGraph(edges=[[0, 0], [0, 1], [1, 1]])
    with pytest.raises(ValueError):
        Graph(non_spatial_graph)


def test_networkx_graph() -> None:
    m = 5
    n = 5
    graph = nx.grid_2d_graph(m=m, n=n)

    mapping = {}
    for i, j in graph.nodes:
        graph.nodes[i, j]['pos'] = (i, j)
        mapping[i, j] = i * m + j

    nx.relabel_nodes(graph, mapping, copy=False)

    layer = Graph(graph)

    assert len(layer.data) == m * n
    assert layer.ndim == 2


def test_networkx_nonspatial_graph() -> None:
    m = 5
    n = 5
    graph = nx.grid_2d_graph(m=m, n=n)

    mapping = {(i, j): i * m + j for i, j in graph.nodes}
    nx.relabel_nodes(graph, mapping, copy=False)

    with pytest.raises(ValueError):
        Graph(graph)


@pytest.mark.parametrize('graph_class', [UndirectedGraph, DirectedGraph])
def test_changing_graph(graph_class: Type[BaseGraph]) -> None:
    graph_a = graph_class(edges=[[0, 1]], coords=[[0, 0], [1, 1]])
    graph_b = graph_class(coords=[[0, 0, 0]])
    layer = Graph(graph_a)
    assert len(layer.data) == graph_a.n_nodes
    assert layer.ndim == graph_a.ndim
    layer.data = graph_b
    assert len(layer.data) == graph_b.n_nodes
    assert layer.ndim == graph_b.ndim


@pytest.mark.parametrize('graph_class', [UndirectedGraph, DirectedGraph])
def test_move(graph_class: Type[BaseGraph]) -> None:
    start_coords = np.asarray([[0, 0], [1, 1], [2, 2]])
    graph = graph_class(edges=[[0, 1], [1, 2]], coords=start_coords)

    layer = Graph(graph)
    assert len(layer.data) == len(start_coords)

    # move one points relative to initial drag start location
    layer._move([0], [0, 0])
    layer._move([0], [10, 10])
    layer._drag_start = None

    assert np.all(layer._points_data[0] == start_coords[0] + [10, 10])
    assert np.all(layer._points_data[1:] == start_coords[1:])

    # move other two points
    layer._move([1, 2], [2, 2])
    layer._move([1, 2], np.add([2, 2], [-3, 4]))
    assert np.all(layer._points_data[0] == start_coords[0] + [10, 10])
    assert np.all(layer._points_data[1:2] == start_coords[1:2] + [-3, 4])


@pytest.mark.parametrize('graph_class', [UndirectedGraph, DirectedGraph])
def test_add_nodes(graph_class: Type[BaseGraph]) -> None:
    # it also tests if original graph object is changed inplace.
    coords = np.asarray([[0, 0], [1, 1]])

    graph = graph_class(edges=[[0, 1]], coords=coords)
    layer = Graph(graph)

    assert len(layer.data) == coords.shape[0]

    # adding without indexing
    layer.add([2, 2])
    assert len(layer.data) == coords.shape[0] + 1
    assert graph.n_nodes == coords.shape[0] + 1
    assert set(layer.selected_data) == {2}

    # adding with index
    layer.add([3, 3], 13)
    assert len(layer.data) == coords.shape[0] + 2
    assert graph.n_nodes == coords.shape[0] + 2
    # buffer index not node index
    assert set(layer.selected_data) == {3}

    # adding multiple with indices
    layer.add([[4, 4], [5, 5]], [24, 25])
    assert len(layer.data) == coords.shape[0] + 4
    assert graph.n_nodes == coords.shape[0] + 4
    assert set(layer.selected_data) == {4, 5}


@pytest.mark.parametrize('graph_class', [UndirectedGraph, DirectedGraph])
def test_remove_selected_nodes(graph_class: Type[BaseGraph]) -> None:
    # it also tests if original graph object is changed inplace.
    coords = np.asarray([[0, 0], [1, 1], [2, 2]])

    graph = graph_class(edges=[[0, 1], [1, 2]], coords=coords)
    layer = Graph(graph)

    # With nothing selected no points should be removed
    layer.remove_selected()
    assert len(layer.data) == coords.shape[0]
    assert graph.n_nodes == coords.shape[0]

    # select nodes and remove then
    layer.selected_data = {0, 2}
    layer.remove_selected()
    assert len(layer.data) == coords.shape[0] - 2
    assert graph.n_nodes == coords.shape[0] - 2

    # on this test, coordinates match index
    assert np.all(graph.get_coordinates() == 1)

    # remove last nodes, note that node id is not zero
    layer.selected_data = {1}
    layer.remove_selected()
    assert len(layer.data) == 0
    assert graph.n_nodes == 0


@pytest.mark.parametrize('graph_class', [UndirectedGraph, DirectedGraph])
def test_remove_nodes(graph_class: Type[BaseGraph]) -> None:
    # it also tests if original graph object is changed inplace.
    coords = np.asarray([[0, 0], [1, 1], [2, 2]])

    graph = graph_class(edges=[[0, 1], [1, 2]], coords=coords)
    layer = Graph(graph)

    # note that their index doesn't change with removals
    layer.remove(1)
    assert len(layer.data) == coords.shape[0] - 1
    assert graph.n_nodes == coords.shape[0] - 1

    # on this test, coordinates match index
    assert not np.any(graph.get_coordinates() == 1)

    layer.remove([0, 2])
    assert len(layer.data) == 0
    assert graph.n_nodes == 0


@pytest.mark.parametrize('graph_class', [UndirectedGraph, DirectedGraph])
def test_remove_nodes_non_sequential_indexing(
    graph_class: Type[BaseGraph],
) -> None:
    # it also tests if original graph object is changed inplace.
    indices = np.asarray([5, 3, 1])
    coords = np.asarray([[0, 0], [1, 1], [2, 2]])
    coords = pd.DataFrame(coords, index=indices)

    graph = graph_class(edges=[[5, 3], [3, 1]], coords=coords)
    layer = Graph(graph)

    # note that their index doesn't change with removals
    layer.remove(indices[1])
    assert len(layer.data) == coords.shape[0] - 1
    assert graph.n_nodes == coords.shape[0] - 1

    # node 3 (remove above) coordinates are all 1
    assert not np.any(graph.get_coordinates() == 1)

    layer.remove(indices[[0, 2]])
    assert len(layer.data) == 0
    assert graph.n_nodes == 0


@pytest.mark.parametrize('graph_class', [UndirectedGraph, DirectedGraph])
def test_graph_out_of_slice_display(graph_class: Type[BaseGraph]) -> None:
    coords = np.asarray([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])

    graph = graph_class(edges=[[0, 1], [1, 2]], coords=coords)
    layer = Graph(graph, out_of_slice_display=True)
    assert layer.out_of_slice_display


def test_graph_from_data_tuple() -> None:
    layer = Graph(name='graph')
    new_layer = Graph.create(*layer.as_layer_data_tuple())
    assert layer.name == new_layer.name
    assert len(layer.data) == len(new_layer.data)
    assert layer.ndim == new_layer.ndim


@pytest.mark.parametrize('graph_class', [UndirectedGraph, DirectedGraph])
def test_graph_from_data_tuple_non_empty(graph_class: Type[BaseGraph]) -> None:
    indices = np.asarray([5, 3, 1])
    coords = np.asarray([[0, 0], [1, 1], [2, 2]])
    coords = pd.DataFrame(coords, index=indices)

    graph = graph_class(edges=[[5, 3], [3, 1]], coords=coords)
    layer = Graph(graph)

    new_layer = Graph.create(*layer.as_layer_data_tuple())
    assert layer.name == new_layer.name
    assert len(layer.data) == len(new_layer.data)
    assert layer.ndim == new_layer.ndim


@pytest.mark.parametrize('graph_class', [UndirectedGraph, DirectedGraph])
def test_add_nodes_buffer_resize(graph_class):
    coords = np.asarray([(0, 0, 0)])

    graph = graph_class(coords=coords)
    layer = Graph(graph, out_of_slice_display=True)

    # adding will cause buffer to resize
    layer.add([(1, 1, 1)])
    assert len(layer.data) == coords.shape[0] + 1
    assert graph.n_nodes == coords.shape[0] + 1


@pytest.mark.parametrize('graph_class', [UndirectedGraph, DirectedGraph])
def test_add_data_event(graph_class):
    coords = np.asarray([[0, 0], [1, 1]])

    graph = graph_class(edges=[[0, 1]], coords=coords)
    layer = Graph(graph)
    layer.events.data = Mock()

    layer.add([5, 5])
    calls = layer.events.data.call_args_list
    assert len(calls) == 2

    first_call = calls[0]
    assert first_call[1]['action'] == ActionType.ADDING
    assert len(first_call[1]['data_indices']) == 1
    # 3rd node added at index 2
    assert first_call[1]['data_indices'] == (2,)

    second_call = calls[1]
    assert second_call[1]['action'] == ActionType.ADDED
    assert second_call[1]['data_indices'] == (2,)

    # specifying index
    new_node = [3, 3]
    layer.add(new_node, [7])
    last_call = calls[-1]
    assert last_call[1]['data_indices'] == (7,)

    # adding multiple at once
    new_nodes = [[4, 4], [5, 5]]
    layer.add(new_nodes)
    last_call = calls[-1]
    assert last_call[1]['data_indices'] == (8, 9)


@pytest.mark.parametrize('graph_class', [UndirectedGraph, DirectedGraph])
def test_remove_data_event(graph_class):
    coords = np.asarray([[0, 0], [1, 1], [2, 2], [3, 3]])

    graph = graph_class(edges=[[0, 1], [1, 2]], coords=coords)
    layer = Graph(graph)
    layer.events.data = Mock()

    layer.remove(1)
    calls = layer.events.data.call_args_list
    assert len(calls) == 2

    first_call = calls[0]
    assert first_call[1]['action'] == ActionType.REMOVING
    assert len(first_call[1]['data_indices']) == 1
    assert first_call[1]['data_indices'] == (1,)

    second_call = calls[1]
    assert second_call[1]['action'] == ActionType.REMOVED
    assert second_call[1]['data_indices'] == (1,)

    # make sure data indices are old values
    layer.remove([2, 3])
    last_call = calls[-1]
    assert last_call[1]['data_indices'] == (2, 3)


@pytest.mark.parametrize('graph_class', [UndirectedGraph, DirectedGraph])
def test_remove_selected_data_event(graph_class):
    coords = np.asarray([[0, 0], [1, 1], [2, 2], [3, 3]])

    graph = graph_class(edges=[[0, 1], [1, 2]], coords=coords)
    layer = Graph(graph)
    layer.events.data = Mock()

    layer.selected_data = {0}
    layer.remove_selected()

    calls = layer.events.data.call_args_list
    assert len(calls) == 2

    first_call = calls[0]
    assert first_call[1]['action'] == ActionType.REMOVING
    assert len(first_call[1]['data_indices']) == 1
    assert first_call[1]['data_indices'] == (0,)

    second_call = calls[1]
    assert second_call[1]['action'] == ActionType.REMOVED
    assert second_call[1]['data_indices'] == (0,)

    layer.selected_data = {1, 2}
    layer.remove_selected()
    last_call = calls[-1]
    assert last_call[1]['data_indices'] == (1, 2)

    # refresh layer to make index reasoning more straightforward
    graph = graph_class(edges=[[0, 1], [1, 2]], coords=coords)
    layer = Graph(graph)
    layer.events.data = Mock()

    layer.add([1, 1], 7)

    # buffer index, not node id
    layer.selected_data = {4}
    layer.remove_selected()
    calls = layer.events.data.call_args_list

    # selected_data uses buffer id, events will always emit world id
    last_call = calls[-1]
    assert last_call[1]['data_indices'] == (7,)
