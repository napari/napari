import numpy as np
import pytest
from numpy import array

from napari.layers.shapes._shapes_utils import (
    generate_2D_edge_meshes,
    get_default_shape_type,
    number_of_shapes,
)

W_DATA = [[0, 3], [1, 0], [2, 3], [5, 0], [2.5, 5]]


def _regen_testcases():
    """
    In case the below test cases need to be update here
    is a simple function you can run to regenerate the `cases` variable below.
    """
    exec(
        """
from napari.layers.shapes._tests.test_shapes_utils import (
    generate_2D_edge_meshes,
    W_DATA,
)


mesh_cases = [
    (W_DATA, False, 3, False),
    (W_DATA, True, 3, False),
    (W_DATA, False, 3, True),
    (W_DATA, True, 3, True),
]


s = '['
for args in mesh_cases:
    cot = generate_2D_edge_meshes(*args)
    s = s + str(['W_DATA', *args[1:], cot]) + ','
s += ']'
s = s.replace("'W_DATA'", 'W_DATA')
print(s)
"""
    )


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
                    [2.0, 3.0],
                    [2.0, 3.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [2.5, 5.0],
                    [2.5, 5.0],
                    [1.0, 0.0],
                    [5.0, 0.0],
                ]
            ),
            array(
                [
                    [0.47434165, 0.15811388],
                    [-0.47434165, -0.15811388],
                    [-0.0, 1.58113883],
                    [-0.47434165, -0.15811388],
                    [-0.21850801, 0.92561479],
                    [0.21850801, -0.92561479],
                    [-1.82514077, 2.53224755],
                    [-0.35355339, -0.35355339],
                    [-0.4472136, -0.2236068],
                    [0.4472136, 0.2236068],
                    [0.47434165, -0.15811388],
                    [0.4472136, 0.2236068],
                ]
            ),
            array(
                [
                    [0, 1, 3],
                    [0, 3, 2],
                    [2, 10, 5],
                    [2, 5, 4],
                    [4, 5, 7],
                    [4, 7, 6],
                    [6, 11, 9],
                    [6, 9, 8],
                    [10, 2, 3],
                    [11, 6, 7],
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
                    [2.0, 3.0],
                    [2.0, 3.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [2.5, 5.0],
                    [2.5, 5.0],
                    [0.0, 3.0],
                    [0.0, 3.0],
                    [1.0, 0.0],
                    [5.0, 0.0],
                ]
            ),
            array(
                [
                    [0.58459244, -0.17263848],
                    [-0.58459244, 0.17263848],
                    [-0.0, 1.58113883],
                    [-0.47434165, -0.15811388],
                    [-0.21850801, 0.92561479],
                    [0.21850801, -0.92561479],
                    [-1.82514077, 2.53224755],
                    [-0.35355339, -0.35355339],
                    [-0.17061484, -0.7768043],
                    [0.17061484, 0.7768043],
                    [0.58459244, -0.17263848],
                    [-0.58459244, 0.17263848],
                    [0.47434165, -0.15811388],
                    [0.4472136, 0.2236068],
                ]
            ),
            array(
                [
                    [0, 1, 3],
                    [0, 3, 2],
                    [2, 12, 5],
                    [2, 5, 4],
                    [4, 5, 7],
                    [4, 7, 6],
                    [6, 13, 9],
                    [6, 9, 8],
                    [8, 9, 11],
                    [8, 11, 10],
                    [12, 2, 3],
                    [13, 6, 7],
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
                    [2.0, 3.0],
                    [2.0, 3.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [2.5, 5.0],
                    [2.5, 5.0],
                    [0.0, 3.0],
                    [1.0, 0.0],
                    [2.0, 3.0],
                    [5.0, 0.0],
                ]
            ),
            array(
                [
                    [0.47434165, 0.15811388],
                    [-0.47434165, -0.15811388],
                    [-0.0, 1.58113883],
                    [-0.47434165, -0.15811388],
                    [-0.47434165, 0.15811388],
                    [0.21850801, -0.92561479],
                    [-1.82514077, 2.53224755],
                    [-0.35355339, -0.35355339],
                    [-0.4472136, -0.2236068],
                    [0.4472136, 0.2236068],
                    [0.47434165, 0.15811388],
                    [0.47434165, -0.15811388],
                    [0.35355339, 0.35355339],
                    [0.4472136, 0.2236068],
                ]
            ),
            array(
                [
                    [10, 1, 3],
                    [10, 3, 2],
                    [2, 11, 5],
                    [2, 5, 4],
                    [12, 5, 7],
                    [12, 7, 6],
                    [6, 13, 9],
                    [6, 9, 8],
                    [0, 1, 10],
                    [11, 2, 3],
                    [4, 5, 12],
                    [13, 6, 7],
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
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 3.0],
                    [2.0, 3.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [2.5, 5.0],
                    [2.5, 5.0],
                    [0.0, 3.0],
                    [0.0, 3.0],
                    [0.0, 3.0],
                    [1.0, 0.0],
                    [2.0, 3.0],
                    [5.0, 0.0],
                    [2.5, 5.0],
                ]
            ),
            array(
                [
                    [0.58459244, -0.17263848],
                    [-0.31234752, 0.3904344],
                    [-0.0, 1.58113883],
                    [-0.47434165, -0.15811388],
                    [-0.47434165, 0.15811388],
                    [0.21850801, -0.92561479],
                    [-1.82514077, 2.53224755],
                    [-0.35355339, -0.35355339],
                    [-0.17061484, -0.7768043],
                    [0.4472136, 0.2236068],
                    [0.58459244, -0.17263848],
                    [-0.31234752, 0.3904344],
                    [-0.47434165, -0.15811388],
                    [0.47434165, -0.15811388],
                    [0.35355339, 0.35355339],
                    [0.4472136, 0.2236068],
                    [-0.31234752, 0.3904344],
                ]
            ),
            array(
                [
                    [0, 12, 3],
                    [0, 3, 2],
                    [2, 13, 5],
                    [2, 5, 4],
                    [14, 5, 7],
                    [14, 7, 6],
                    [6, 15, 9],
                    [6, 9, 8],
                    [8, 16, 11],
                    [8, 11, 10],
                    [12, 0, 1],
                    [13, 2, 3],
                    [4, 5, 14],
                    [15, 6, 7],
                    [16, 8, 9],
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
