import sys

import numpy as np
import numpy.testing as npt
import pytest
from vispy.geometry import PolygonData

from napari.layers.shapes._shapes_models import (
    Ellipse,
    Line,
    Path,
    Polygon,
    Rectangle,
)
from napari.layers.shapes._shapes_utils import triangulate_face

BETTER_TRIANGULATION = (
    'triangle' in sys.modules or 'PartSegCore_compiled_backend' in sys.modules
)


def test_rectangle1():
    """Test creating Rectangle by four corners."""
    np.random.seed(0)
    data = np.array([(10, 10), (20, 10), (20, 20), (10, 20)], dtype=np.float32)
    shape = Rectangle(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 0)


def test_rectangle2():
    """Test creating Rectangle by upper left and bottom right."""
    # If given two corners, representation will be expanded to four
    data = np.array([[-10, -10], [20, 20]], dtype=np.float32)
    shape = Rectangle(data)
    assert len(shape.data) == 4
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 0)


def test_rectangle_bounding_box():
    """Test that the bounding box is correctly updated based on edge width."""
    data = [[10, 10], [20, 20]]
    shape = Rectangle(data)
    npt.assert_array_equal(
        shape.bounding_box, np.array([[9.5, 9.5], [20.5, 20.5]])
    )
    shape.edge_width = 2
    npt.assert_array_equal(shape.bounding_box, np.array([[9, 9], [21, 21]]))
    shape.edge_width = 4
    npt.assert_array_equal(shape.bounding_box, np.array([[8, 8], [22, 22]]))


def test_rectangle_shift():
    shape = Rectangle(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    npt.assert_array_equal(
        shape.bounding_box, np.array([[-0.5, -0.5], [1.5, 1.5]])
    )

    shape.shift((1, 1))
    npt.assert_array_equal(
        shape.data, np.array([[1, 1], [2, 1], [2, 2], [1, 2]])
    )
    npt.assert_array_equal(
        shape.bounding_box, np.array([[0.5, 0.5], [2.5, 2.5]])
    )


def test_rectangle_rotate():
    shape = Rectangle(np.array([[1, 2], [-1, 2], [-1, -2], [1, -2]]))
    npt.assert_array_equal(
        shape.bounding_box, np.array([[-1.5, -2.5], [1.5, 2.5]])
    )
    shape.rotate(-90)
    npt.assert_array_almost_equal(
        shape.data, np.array([[-2, 1], [-2, -1], [2, -1], [2, 1]])
    )
    npt.assert_array_almost_equal(
        shape.bounding_box, np.array([[-2.5, -1.5], [2.5, 1.5]])
    )


def test_nD_rectangle():
    """Test creating Shape with a single four corner planar 3D rectangle."""
    data = np.array(
        [[0, -10, -10], [0, -10, 20], [0, 20, 20], [0, 20, -10]],
        dtype=np.float32,
    )
    shape = Rectangle(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 1)

    shape.ndisplay = 3
    assert shape.data_displayed.shape == (4, 3)


def test_polygon_data_triangle():
    data = np.array(
        [
            [10.97627008, 14.30378733],
            [12.05526752, 10.89766366],
            [8.47309599, 12.91788226],
            [8.75174423, 17.83546002],
            [19.27325521, 7.66883038],
            [15.83450076, 10.5778984],
        ]
    )
    vertices, _triangles = PolygonData(vertices=data).triangulate()

    assert vertices.shape == (8, 2)


def test_polygon_data_triangle_module():
    pytest.importorskip('triangle')
    data = np.array(
        [
            [10.97627008, 14.30378733],
            [12.05526752, 10.89766366],
            [8.47309599, 12.91788226],
            [8.75174423, 17.83546002],
            [19.27325521, 7.66883038],
            [15.83450076, 10.5778984],
        ]
    )
    vertices, _triangles = triangulate_face(data)

    assert vertices.shape == (6, 2)


def test_polygon():
    """Test creating Shape with a random polygon."""
    # Test a single non convex six vertex polygon
    data = np.array(
        [
            [10.97627008, 14.30378733],
            [12.05526752, 10.89766366],
            [8.47309599, 12.91788226],
            [8.75174423, 17.83546002],
            [19.27325521, 7.66883038],
            [15.83450076, 10.5778984],
        ],
        dtype=np.float32,
    )
    shape = Polygon(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (6, 2)
    assert shape.slice_key.shape == (2, 0)
    # should get few triangles
    expected_face = (6, 2) if BETTER_TRIANGULATION else (8, 2)
    assert shape._edge_vertices.shape == (16, 2)
    assert shape._face_vertices.shape == expected_face


def test_polygon2():
    data = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    shape = Polygon(data, interpolation_order=3)
    # should get many triangles

    expected_face = (249, 2)

    assert shape._edge_vertices.shape == (500, 2)
    assert shape._face_vertices.shape == expected_face


def test_polygon3():
    data = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]])
    shape = Polygon(data, interpolation_order=3, ndisplay=3)
    # should get many vertices
    assert shape._edge_vertices.shape == (2500, 3)
    # faces are not made for non-coplanar 3d stuff
    assert shape._face_vertices.shape == (0, 3)


def test_nD_polygon():
    """Test creating Shape with a random nD polygon."""
    # Test a single six vertex planar 3D polygon
    np.random.seed(0)
    data = 20 * np.random.random((6, 3)).astype(np.float32)
    data[:, 0] = 0
    shape = Polygon(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (6, 2)
    assert shape.slice_key.shape == (2, 1)

    shape.ndisplay = 3
    assert shape.data_displayed.shape == (6, 3)


def test_path():
    """Test creating Shape with a random path."""
    # Test a single six vertex path
    np.random.seed(0)
    data = 20 * np.random.random((6, 2)).astype(np.float32)
    shape = Path(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (6, 2)
    assert shape.slice_key.shape == (2, 0)


def test_nD_path():
    """Test creating Shape with a random nD path."""
    # Test a single six vertex 3D path
    np.random.seed(0)
    data = 20 * np.random.random((6, 3)).astype(np.float32)
    shape = Path(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (6, 2)
    assert shape.slice_key.shape == (2, 1)

    shape.ndisplay = 3
    assert shape.data_displayed.shape == (6, 3)


def test_line():
    """Test creating 2D Line."""
    np.random.seed(0)
    data = np.array([[10, 10], [20, 20]], dtype=np.float32)
    shape = Line(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (2, 2)
    assert shape.slice_key.shape == (2, 0)


def test_nD_line():
    """Test creating Line in 3d"""
    data = np.array([[10, 10, 10], [20, 20, 20]], dtype=np.float32)
    shape = Line(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (2, 2)
    assert shape.slice_key.shape == (2, 1)

    shape.ndisplay = 3
    assert shape.data_displayed.shape == (2, 3)


def test_ellipse1():
    """Test creating Ellipse by four corners."""
    data = np.array([(10, 10), (20, 10), (20, 20), (10, 20)], dtype=np.float32)
    shape = Ellipse(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 0)


def test_ellipse2():
    """Test creating Ellipse by upper left and lower right corners."""
    data = np.array([[10, 10], [20, 20]], dtype=np.float32)
    shape = Ellipse(data)
    assert len(shape.data) == 4
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 0)


def test_nD_ellipse():
    """Test creating Shape with a random nD ellipse."""
    # Test a single four corner planar 3D ellipse
    data = np.array(
        [(0, -10, -10), (0, -10, 20), (0, 20, 20), (0, 20, -10)],
        dtype=np.float32,
    )
    shape = Ellipse(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 1)

    shape.ndisplay = 3
    assert shape.data_displayed.shape == (4, 3)


def test_ellipse_shift():
    shape = Ellipse(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    npt.assert_array_equal(
        shape.bounding_box, np.array([[-0.5, -0.5], [1.5, 1.5]])
    )

    shape.shift((1, 1))
    npt.assert_array_equal(
        shape.data, np.array([[1, 1], [2, 1], [2, 2], [1, 2]])
    )
    npt.assert_array_equal(
        shape.bounding_box, np.array([[0.5, 0.5], [2.5, 2.5]])
    )


def test_ellipse_rotate():
    shape = Ellipse(np.array([[1, 2], [-1, 2], [-1, -2], [1, -2]]))
    npt.assert_array_equal(
        shape.bounding_box, np.array([[-1.5, -2.5], [1.5, 2.5]])
    )
    shape.rotate(-90)
    npt.assert_array_almost_equal(
        shape.data, np.array([[-2, 1], [-2, -1], [2, -1], [2, 1]])
    )
    npt.assert_array_almost_equal(
        shape.bounding_box, np.array([[-2.5, -1.5], [2.5, 1.5]])
    )
