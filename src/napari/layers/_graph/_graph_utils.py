from networkx import Graph as nxGraph


def fix_graph_data(
    graph: nxGraph | None = None, ndim: int | None = None
) -> tuple[nxGraph, int]:
    """
    Ensure that graph data is valid and return the graph and its ndim.

    Parameters
    ----------
    graph : networkx.Graph or None
        Graph to be checked
    ndim : int or None
        number of expected dimensions

    Returns
    -------
    graph : networkx.Graph
        Graph data
    ndim : int
        number of dimensions

    Raises
    ------
    TypeError
        if graph is not a networkx.Graph or None
    ValueError
        if ndim does not match with the dimensions of the graph nodes
    """

    if graph is None:
        graph = nxGraph()
    elif not isinstance(graph, nxGraph):
        raise TypeError('Data must be a networkx.Graph or None')

    nodes = list(graph.nodes(data=True))

    if nodes:
        if not all('coordinates' in node_attr for _, node_attr in nodes):
            raise ValueError('All nodes need to have coordinates')
        if len({len(node[1]['coordinates']) for node in nodes}) != 1:
            raise ValueError('All node coordinates need to have the same ndim')

    # Determine the dimensionality of the graph
    first_node = next(iter(graph.nodes(data=True)), None)
    ndim_ = (
        len(first_node[1]['coordinates'])
        if first_node
        else (2 if ndim is None else ndim)
    )

    if ndim is not None and ndim_ != ndim:
        raise ValueError(
            f'The graph nodes dimensions {ndim_} do not match the provided ndim argument {ndim}'
        )

    return graph, ndim_
