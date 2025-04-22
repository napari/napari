import importlib
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest

from napari.layers.shapes import (
    _accelerated_triangulate_python,
)

ac = pytest.importorskip('napari.layers.shapes._accelerated_triangulate_numba')


@pytest.fixture(params=[False, True])
def _disable_jit(request):
    """Fixture to temporarily disable numba JIT during testing.

    This helps to measure coverage and in debugging. *However*, reloading a
    module can cause issues with object instance / class relationships, so
    the `_accelerated_cmap` module should be as small as possible and contain
    no class definitions, only functions.
    """
    pytest.importorskip('numba')
    with patch('numba.core.config.DISABLE_JIT', request.param):
        importlib.reload(ac)
        yield
    importlib.reload(ac)


@pytest.mark.parametrize(
    ('path', 'closed', 'bevel', 'expected'),
    [
        ([[0, 0], [0, 10], [10, 10], [10, 0]], True, False, 10),
        ([[0, 0], [0, 10], [10, 10], [10, 0]], False, False, 8),
        ([[0, 0], [0, 10], [10, 10], [10, 0]], True, True, 14),
        ([[0, 0], [0, 10], [10, 10], [10, 0]], False, True, 10),
        ([[2, 10], [0, -5], [-2, 10], [-2, -10], [2, -10]], True, False, 15),
        ([[0, 0], [0, 10]], False, False, 4),
        ([[0, 0], [0, 10], [0, 20]], False, False, 6),
        ([[0, 0], [0, 2], [10, 1]], True, False, 9),
        ([[0, 0], [10, 1], [9, 1.1]], False, False, 7),
        ([[9, 0.9], [10, 1], [0, 2]], False, False, 7),
        ([[0, 0], [-10, 1], [-9, 1.1]], False, False, 7),
        ([[-9, 0.9], [-10, 1], [0, 2]], False, False, 7),
    ],
)
@pytest.mark.usefixtures('_disable_jit')
def test_generate_2D_edge_meshes(path, closed, bevel, expected):
    centers, offsets, triangles = ac.generate_2D_edge_meshes(
        np.array(path, dtype='float32'), closed=closed, bevel=bevel
    )
    assert centers.shape == offsets.shape
    assert centers.shape[0] == expected
    assert triangles.shape[0] == expected - 2


@pytest.mark.parametrize(
    ('data', 'expected', 'closed'),
    [
        (
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            True,
        ),
        (
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            False,
        ),
        (
            np.array(
                [[0, 0], [1, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'
            ),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            True,
        ),
        (
            np.array(
                [[0, 0], [1, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'
            ),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            False,
        ),
        (
            np.array(
                [[0, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 1], [0, 1]],
                dtype='float32',
            ),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            False,
        ),
        (
            np.array(
                [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype='float32'
            ),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            True,
        ),
        (
            np.array(
                [
                    [0, 0],
                    [1, 0],
                    [1, 1],
                    [0, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                dtype='float32',
            ),
            np.array(
                [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype='float32'
            ),
            True,
        ),
    ],
)
@pytest.mark.usefixtures('_disable_jit')
def test_remove_path_duplicates(data, expected, closed):
    result = ac.remove_path_duplicates(data, closed=closed)
    assert np.all(result == expected)


@pytest.mark.usefixtures('_disable_jit')
def test_create_box_from_bounding():
    bounding = np.array([[0, 0], [2, 2]], dtype='float32')
    box = ac.create_box_from_bounding(bounding)
    assert box.shape == (9, 2)
    npt.assert_array_equal(
        box,
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [2, 1],
            [2, 2],
            [1, 2],
            [0, 2],
            [0, 1],
            [1, 1],
        ],
    )


@pytest.mark.usefixtures('_disable_jit')
def test_is_convex_self_intersection(self_intersecting_polygon):
    assert not ac.is_convex(self_intersecting_polygon)


@pytest.mark.usefixtures('_disable_jit')
def test_is_convex_regular_polygon(regular_polygon):
    assert ac.is_convex(regular_polygon)


@pytest.mark.usefixtures('_disable_jit')
def test_is_convex_non_convex(non_convex_poly):
    assert not ac.is_convex(non_convex_poly)


@pytest.mark.usefixtures('_disable_jit')
def test_line_non_convex(line):
    assert not ac.is_convex(line)


@pytest.mark.usefixtures('_disable_jit')
def test_line_two_point_non_convex(line_two_point):
    assert not ac.is_convex(line_two_point)


@pytest.mark.usefixtures('_disable_jit')
def test_normalize_vertices_and_edges(poly_hole):
    points, edges = ac.normalize_vertices_and_edges(poly_hole, close=True)
    assert points.shape == (8, 2)
    assert edges.shape == (8, 2)


@pytest.mark.usefixtures('_disable_jit')
def test_reconstruct_polygon_edges():
    vertices = np.array(
        [(0, 0), (3, 0), (3, 3), (0, 3), (1, 1), (2, 1), (2, 2), (1, 2)]
    )
    edges = np.array(
        [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
    )

    res = ac.reconstruct_polygons_from_edges(vertices, edges)
    assert len(res) == 2
    assert len(res[0]) == 4
    assert len(res[1]) == 4


@pytest.fixture
def country_wth_hole():
    return np.array(
        [
            [-28.58, 196.34],
            [-28.08, 196.82],
            [-28.36, 197.22],
            [-28.78, 197.39],
            [-28.86, 197.84],
            [-29.05, 198.46],
            [-28.97, 199.0],
            [-28.46, 199.89],
            [-24.77, 199.9],
            [-24.92, 200.17],
            [-25.87, 200.76],
            [-26.48, 200.67],
            [-26.83, 200.89],
            [-26.73, 201.61],
            [-26.28, 202.11],
            [-25.98, 202.58],
            [-25.5, 202.82],
            [-25.27, 203.31],
            [-25.39, 203.73],
            [-25.67, 204.21],
            [-25.72, 205.03],
            [-25.49, 205.66],
            [-25.17, 205.77],
            [-24.7, 205.94],
            [-24.62, 206.49],
            [-24.24, 206.79],
            [-23.57, 207.12],
            [-22.83, 208.02],
            [-22.09, 209.43],
            [-22.1, 209.84],
            [-22.27, 210.32],
            [-22.15, 210.66],
            [-22.25, 211.19],
            [-23.66, 211.67],
            [-24.37, 211.93],
            [-25.48, 211.75],
            [-25.84, 211.84],
            [-25.66, 211.33],
            [-25.73, 211.04],
            [-26.02, 210.95],
            [-26.4, 210.68],
            [-26.74, 210.69],
            [-27.29, 211.28],
            [-27.18, 211.87],
            [-26.73, 212.07],
            [-26.74, 212.83],
            [-27.47, 212.58],
            [-28.3, 212.46],
            [-28.75, 212.2],
            [-29.26, 211.52],
            [-29.4, 211.33],
            [-29.91, 210.9],
            [-30.42, 210.62],
            [-31.14, 210.06],
            [-32.17, 208.93],
            [-32.77, 208.22],
            [-33.23, 207.46],
            [-33.61, 206.42],
            [-33.67, 205.91],
            [-33.94, 205.78],
            [-33.8, 205.17],
            [-33.99, 204.68],
            [-33.79, 203.59],
            [-33.92, 202.99],
            [-33.86, 202.57],
            [-34.26, 201.54],
            [-34.42, 200.69],
            [-34.8, 200.07],
            [-34.82, 199.62],
            [-34.46, 199.19],
            [-34.44, 198.86],
            [-34.0, 198.42],
            [-34.14, 198.38],
            [-33.87, 198.24],
            [-33.28, 198.25],
            [-32.61, 197.93],
            [-32.43, 198.25],
            [-31.66, 198.22],
            [-30.73, 197.57],
            [-29.88, 197.06],
            [-28.58, 196.34],
            [-28.96, 208.98],
            [-28.65, 208.54],
            [-28.85, 208.07],
            [-29.24, 207.53],
            [-29.88, 207.0],
            [-30.65, 207.75],
            [-30.55, 208.11],
            [-30.23, 208.29],
            [-30.07, 208.85],
            [-29.74, 209.02],
            [-29.26, 209.33],
            [-28.96, 208.98],
        ]
    )


def test_normalize_vertices_and_edges_py_numba_same(country_wth_hole):
    v1, e1 = ac.normalize_vertices_and_edges(country_wth_hole, close=True)
    v2, e2 = _accelerated_triangulate_python.normalize_vertices_and_edges_py(
        country_wth_hole, close=True
    )
    e1s = {tuple(x) for x in e1}
    e2s = {tuple(x) for x in e2}
    assert e1s == e2s
    npt.assert_array_equal(v1, v2)
