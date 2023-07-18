"""
Add graph
===================

Display a random undirected graph using the graph layer.

.. tags:: visualization-basic
"""

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
    graph = UndirectedGraph(edges=edges, coords=nodes_df)

    return graph


graph = build_graph(n_nodes=1_000_000, n_neighbors=5)

viewer = napari.Viewer()
layer = viewer.add_graph(graph, out_of_slice_display=True)


if __name__ == "__main__":

    napari.run()
