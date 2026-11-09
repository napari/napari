"""
Add graph 3D
============

Display a graph layer in 3D using the ``add_graph`` API.

.. tags:: visualization-nD
"""

import networkx as nx
import numpy as np

import napari

n_nodes = 100  # nodes per strand
angles = np.linspace(0, 3 * np.pi, n_nodes)  # three turns
z = np.linspace(0, 100, n_nodes)
radius = np.linspace(5, 10, n_nodes)  # narrow bottom, wide top

# Two strands in opposite phase, each node has (z, y, x) coordinates
first_strand_pos = np.column_stack([z, radius * np.sin(angles), radius * np.cos(angles)])
second_strand_pos = np.column_stack([z, radius * np.sin(angles + np.pi), radius * np.cos(angles + np.pi)])
strands_pos = [first_strand_pos, second_strand_pos]

graph = nx.Graph()
graph.add_nodes_from(
    (i, {'coordinates': c}) for i, c in enumerate(np.concatenate(strands_pos))
)
graph.add_edges_from(nx.path_graph(n_nodes).edges)  # first strand
graph.add_edges_from((i + n_nodes, j + n_nodes) for i, j in nx.path_graph(n_nodes).edges)  # second strand
graph.add_edges_from((i, i + n_nodes) for i in range(0, n_nodes, 2))  # rungs

viewer = napari.Viewer(ndisplay=3)
viewer._add_graph(graph)


if __name__ == '__main__':
    napari.run()
