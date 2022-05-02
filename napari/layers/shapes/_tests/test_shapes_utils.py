import numpy as np
import pytest
from numpy import array

from napari.layers.shapes._shapes_utils import (
    generate_2D_edge_meshes,
    get_default_shape_type,
    number_of_shapes,
)

W_DATA = [[0, 3], [1, 0], [2, 3], [5, 0], [2.5, 5]]

cases = [
    [
        W_DATA,
        False,
        3,
        False,
        (
            array(
                [
                    [0.0, 3.0],
                    [0.0, 3.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 3.0],
                    [2.0, 3.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [2.5, 5.0],
                    [2.5, 5.0],
                ]
            ),
            array(
                [
                    [0.47434165, 0.15811388],
                    [-0.47434165, -0.15811388],
                    [-0.5, -0.0],
                    [-0.0, 1.5],
                    [0.5, 0.0],
                    [-0.21850801, 0.92561479],
                    [0.21850801, -0.92561479],
                    [-0.40562109, -0.29235514],
                    [-0.87706543, 1.21686328],
                    [0.40562109, 0.29235514],
                    [-0.4472136, -0.2236068],
                    [0.4472136, 0.2236068],
                ]
            ),
            array(
                [
                    [0, 1, 3],
                    [1, 2, 3],
                    [2, 3, 4],
                    [3, 4, 6],
                    [3, 5, 6],
                    [5, 6, 8],
                    [6, 7, 8],
                    [7, 8, 9],
                    [8, 9, 11],
                    [8, 10, 11],
                ]
            ),
        ),
    ],
    [
        W_DATA,
        True,
        3,
        False,
        (
            array(
                [
                    [0.0, 3.0],
                    [0.0, 3.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 3.0],
                    [2.0, 3.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [2.5, 5.0],
                    [2.5, 5.0],
                ]
            ),
            array(
                [
                    [0.58459244, -0.17263848],
                    [-0.58459244, 0.17263848],
                    [-0.5, -0.0],
                    [-0.0, 1.5],
                    [0.5, 0.0],
                    [-0.21850801, 0.92561479],
                    [0.21850801, -0.92561479],
                    [-0.40562109, -0.29235514],
                    [-0.87706543, 1.21686328],
                    [0.40562109, 0.29235514],
                    [-0.17061484, -0.7768043],
                    [0.17061484, 0.7768043],
                ]
            ),
            array(
                [
                    [0, 1, 3],
                    [1, 2, 3],
                    [2, 3, 4],
                    [3, 4, 6],
                    [3, 5, 6],
                    [5, 6, 8],
                    [6, 7, 8],
                    [7, 8, 9],
                    [8, 9, 11],
                    [8, 10, 11],
                    [10, 11, 1],
                    [10, 0, 1],
                ]
            ),
        ),
    ],
    [
        W_DATA,
        False,
        3,
        True,
        (
            array(
                [
                    [0.0, 3.0],
                    [0.0, 3.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 3.0],
                    [2.0, 3.0],
                    [2.0, 3.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [2.5, 5.0],
                    [2.5, 5.0],
                ]
            ),
            array(
                [
                    [0.47434165, 0.15811388],
                    [-0.47434165, -0.15811388],
                    [-0.5, -0.0],
                    [-0.0, 1.5],
                    [0.5, 0.0],
                    [-0.48662449, -0.11487646],
                    [0.34462938, -1.45987348],
                    [0.48662449, 0.11487646],
                    [-0.40562109, -0.29235514],
                    [-0.87706543, 1.21686328],
                    [0.40562109, 0.29235514],
                    [-0.4472136, -0.2236068],
                    [0.4472136, 0.2236068],
                ]
            ),
            array(
                [
                    [0, 1, 3],
                    [1, 2, 3],
                    [2, 3, 4],
                    [3, 4, 6],
                    [3, 5, 6],
                    [5, 6, 7],
                    [6, 7, 9],
                    [6, 8, 9],
                    [8, 9, 10],
                    [9, 10, 12],
                    [9, 11, 12],
                ]
            ),
        ),
    ],
    [
        W_DATA,
        True,
        3,
        True,
        (
            array(
                [
                    [0.0, 3.0],
                    [0.0, 3.0],
                    [0.0, 3.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 3.0],
                    [2.0, 3.0],
                    [2.0, 3.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [2.5, 5.0],
                    [2.5, 5.0],
                    [2.5, 5.0],
                ]
            ),
            array(
                [
                    [0.14161119, 0.47952713],
                    [1.43858139, -0.42483358],
                    [-0.14161119, -0.47952713],
                    [-0.5, -0.0],
                    [-0.0, 1.5],
                    [0.5, 0.0],
                    [-0.48662449, -0.11487646],
                    [0.34462938, -1.45987348],
                    [0.48662449, 0.11487646],
                    [-0.40562109, -0.29235514],
                    [-0.87706543, 1.21686328],
                    [0.40562109, 0.29235514],
                    [0.48835942, -0.10726172],
                    [-0.32178517, -1.46507826],
                    [-0.48835942, 0.10726172],
                ]
            ),
            array(
                [
                    [0, 1, 2],
                    [1, 2, 4],
                    [2, 3, 4],
                    [3, 4, 5],
                    [4, 5, 7],
                    [4, 6, 7],
                    [6, 7, 8],
                    [7, 8, 10],
                    [7, 9, 10],
                    [9, 10, 11],
                    [10, 11, 13],
                    [11, 12, 13],
                    [12, 13, 14],
                    [13, 14, 1],
                    [14, 0, 1],
                ]
            ),
        ),
    ],
]


@pytest.mark.parametrize(
    'path, closed, limit, bevel, expected',
    cases,
)
def test_generate_2D_edge_meshes(
    path,
    closed,
    limit,
    bevel,
    expected,
):
    pass
    c, o, t = generate_2D_edge_meshes(path, closed, limit, bevel)
    expected_center, expected_offsets, expected_triangles = expected
    assert np.allclose(c, expected_center)
    assert np.allclose(o, expected_offsets)
    assert (t == expected_triangles).all()


def test_no_shapes():
    """Test no shapes."""
    assert number_of_shapes([]) == 0
    assert number_of_shapes(np.empty((0, 4, 2))) == 0


def test_one_shape():
    """Test one shape."""
    assert number_of_shapes(np.random.random((4, 2))) == 1


def test_many_shapes():
    """Test many shapes."""
    assert number_of_shapes(np.random.random((8, 4, 2))) == 8


def test_get_default_shape_type():
    """Test getting default shape type"""
    shape_type = ['polygon', 'polygon']
    assert get_default_shape_type(shape_type) == 'polygon'

    shape_type = []
    assert get_default_shape_type(shape_type) == 'polygon'

    shape_type = ['ellipse', 'rectangle']
    assert get_default_shape_type(shape_type) == 'polygon'

    shape_type = ['rectangle', 'rectangle']
    assert get_default_shape_type(shape_type) == 'rectangle'

    shape_type = ['ellipse', 'ellipse']
    assert get_default_shape_type(shape_type) == 'ellipse'

    shape_type = ['polygon']
    assert get_default_shape_type(shape_type) == 'polygon'
