import sys

import numpy as np
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


def test_rectangle():
    """Test creating Shape with a random rectangle."""
    # Test a single four corner rectangle
    np.random.seed(0)
    data = 20 * np.random.random((4, 2))
    shape = Rectangle(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 0)

    # If given two corners, representation will be exapanded to four
    data = 20 * np.random.random((2, 2))
    shape = Rectangle(data)
    assert len(shape.data) == 4
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 0)


def test_nD_rectangle():
    """Test creating Shape with a random nD rectangle."""
    # Test a single four corner planar 3D rectangle
    np.random.seed(0)
    data = 20 * np.random.random((4, 3))
    data[:, 0] = 0
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
    pytest.importorskip("triangle")
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
    # Test a single six vertex polygon
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
    shape = Polygon(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (6, 2)
    assert shape.slice_key.shape == (2, 0)
    # should get few triangles
    expected_face = (6, 2) if "triangle" in sys.modules else (8, 2)
    assert shape._edge_vertices.shape == (16, 2)
    assert shape._face_vertices.shape == expected_face


def test_polygon2():
    data = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    shape = Polygon(data, interpolation_order=3)
    # should get many triangles

    expected_face = (249, 2) if "triangle" in sys.modules else (251, 2)

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
    data = 20 * np.random.random((6, 3))
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
    data = 20 * np.random.random((6, 2))
    shape = Path(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (6, 2)
    assert shape.slice_key.shape == (2, 0)


def test_nD_path():
    """Test creating Shape with a random nD path."""
    # Test a single six vertex 3D path
    np.random.seed(0)
    data = 20 * np.random.random((6, 3))
    shape = Path(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (6, 2)
    assert shape.slice_key.shape == (2, 1)

    shape.ndisplay = 3
    assert shape.data_displayed.shape == (6, 3)


def test_line():
    """Test creating Shape with a random line."""
    # Test a single two vertex line
    np.random.seed(0)
    data = 20 * np.random.random((2, 2))
    shape = Line(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (2, 2)
    assert shape.slice_key.shape == (2, 0)


def test_nD_line():
    """Test creating Shape with a random nD line."""
    # Test a single two vertex 3D line
    np.random.seed(0)
    data = 20 * np.random.random((2, 3))
    shape = Line(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (2, 2)
    assert shape.slice_key.shape == (2, 1)

    shape.ndisplay = 3
    assert shape.data_displayed.shape == (2, 3)


def test_ellipse():
    """Test creating Shape with a random ellipse."""
    # Test a single four corner ellipse
    np.random.seed(0)
    data = 20 * np.random.random((4, 2))
    shape = Ellipse(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 0)

    # If center radii, representation will be exapanded to four corners
    data = 20 * np.random.random((2, 2))
    shape = Ellipse(data)
    assert len(shape.data) == 4
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 0)


def test_nD_ellipse():
    """Test creating Shape with a random nD ellipse."""
    # Test a single four corner planar 3D ellipse
    np.random.seed(0)
    data = 20 * np.random.random((4, 3))
    data[:, 0] = 0
    shape = Ellipse(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 1)

    shape.ndisplay = 3
    assert shape.data_displayed.shape == (4, 3)
