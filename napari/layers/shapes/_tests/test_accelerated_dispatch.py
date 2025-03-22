import numpy as np

from napari.layers.shapes._accelerated_triangulate_dispatch import (
    reconstruct_polygon_edges,
)


def test_reconstruct_polygon_edges():
    vertices = np.array(
        [(0, 0), (3, 0), (3, 3), (0, 3), (1, 1), (2, 1), (2, 2), (1, 2)]
    )
    edges = np.array(
        [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
    )

    res = reconstruct_polygon_edges(vertices, edges)
    assert len(res) == 2
