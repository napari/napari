import numpy as np
import pandas as pd
from napari_graph import UndirectedGraph

import napari
from napari.layers import Graph


def build_graph(n_nodes: int, n_neighbors: int) -> UndirectedGraph:
    neighbors = np.random.randint(n_nodes, size=(n_nodes * n_neighbors))
    edges = np.stack([np.repeat(np.arange(n_nodes), n_neighbors), neighbors], axis=1)

    nodes_df = pd.DataFrame(
        400 * np.random.uniform(size=(n_nodes, 4)),
        columns=["t", "z", "y", "x"],
    )
    graph = UndirectedGraph(edges=edges, coords=nodes_df[["t", "z", "y", "x"]])

    return graph


if __name__ == "__main__":

    viewer = napari.Viewer()
    n_nodes = 1000000
    graph = build_graph(n_nodes, 5)
    layer = Graph(graph, out_of_slice_display=True)
    viewer.add_layer(layer)

    napari.run()
