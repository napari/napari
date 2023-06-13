"""
Nuclei Segmentation Graph
===============

Creates a delaunay graph from maxima of cell nuclei.

.. tags:: visualization-nD
"""
from itertools import combinations

import numpy as np
from napari_graph import UndirectedGraph
from scipy.spatial import Delaunay
from skimage import data, feature, filters

import napari


def delaunay_edges(points: np.ndarray) -> np.ndarray:
    delaunay = Delaunay(points)
    edges = set()
    for simplex in delaunay.simplices:
        # each simplex is represented as a list of four points.
        # we add all edges between the points to the edge list
        edges |= set(combinations(simplex, 2))

    return np.asarray(list(edges))


cells = data.cells3d()

nuclei = cells[:, 1]
smooth = filters.gaussian(nuclei, sigma=10)
nodes_coords = feature.peak_local_max(smooth)
edges = delaunay_edges(nodes_coords)
graph = UndirectedGraph(edges, nodes_coords)
viewer = napari.view_image(
    cells, channel_axis=1, name=['membranes', 'nuclei'], ndisplay=3
)
viewer.add_graph(graph)
viewer.camera.angles = (10, -20, 130)

if __name__ == '__main__':
    napari.run()
