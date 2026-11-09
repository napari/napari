from networkx import Graph as nxGraph

from napari.layers._graph._graph import _Graph


def test_none_graph():
    graph_layer = _Graph(data=None, ndim=None)
    assert graph_layer.data.number_of_nodes() == 0
    assert graph_layer.ndim == 2


def test_empty_graph():
    graph = nxGraph()
    graph_layer = _Graph(data=graph, ndim=None)
    assert graph_layer.data.number_of_nodes() == 0
    assert graph_layer.ndim == 2


def test_ndim_on_empty_and_none_graph():
    empty_graph = nxGraph()
    graph_layer_empty = _Graph(data=empty_graph, ndim=3)
    assert graph_layer_empty.ndim == 3

    none_graph_layer = _Graph(data=None, ndim=4)
    assert none_graph_layer.ndim == 4
