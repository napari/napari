"""
Add graph
===================

Display a random undirected graph using the graph layer.

.. tags:: visualization-basic
"""

import pandas as pd
from napari_graph import UndirectedGraph

import napari


def build_graph(n_nodes: int, n_neighbors: int) -> UndirectedGraph:
    edges = [[0, 2], [3, 4]]

    nodes_df = pd.DataFrame(
        {
            't': [0, 1, 2, 3, 4],
            'y': [0, 20, 45, 70, 90],
            'x': [0, 20, 45, 70, 90]
        }
    )
    graph = UndirectedGraph(edges=edges, coords=nodes_df)

    return graph


graph = build_graph(n_nodes=100, n_neighbors=5)

viewer = napari.Viewer()
layer = viewer.add_graph(graph, out_of_slice_display=True, size=5, projection_mode='all')


if __name__ == "__main__":

    napari.run()
