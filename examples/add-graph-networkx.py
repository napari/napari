"""
Add networkx graph
==================

Add a networkx graph directly to napari. This works as long as nodes
have a "pos" attribute with the node coordinate.

.. tags:: visualization-basic
"""

import networkx as nx

import napari

hex_grid = nx.hexagonal_lattice_graph(5, 5, with_positions=True)
# below conversion not needed after napari/napari-graph#11 is released
hex_grid_ints = nx.convert_node_labels_to_integers(hex_grid)

viewer = napari.Viewer()
layer = viewer.add_graph(hex_grid_ints, size=1)

if __name__ == "__main__":
    napari.run()
