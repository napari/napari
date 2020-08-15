import numpy as np

from napari.layers.shapes._shapes_models import (
    Ellipse,
    Line,
    Path,
    Polygon,
    Rectangle,
)


def test_rectangle():
    """Test creating Shape with a random rectangle."""
    # Test a single four corner rectangle
    np.random.seed(0)
    data = 20 * np.random.random((4, 2))
    shape = Rectangle(data)
    assert np.all(shape.data == data)
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
    assert np.all(shape.data == data)
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 1)

    shape.ndisplay = 3
    assert shape.data_displayed.shape == (4, 3)


def test_polygon():
    """Test creating Shape with a random polygon."""
    # Test a single six vertex polygon
    np.random.seed(0)
    data = 20 * np.random.random((6, 2))
    shape = Polygon(data)
    assert np.all(shape.data == data)
    assert shape.data_displayed.shape == (6, 2)
    assert shape.slice_key.shape == (2, 0)


def test_nD_polygon():
    """Test creating Shape with a random nD polygon."""
    # Test a single six vertex planar 3D polygon
    np.random.seed(0)
    data = 20 * np.random.random((6, 3))
    data[:, 0] = 0
    shape = Polygon(data)
    assert np.all(shape.data == data)
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
    assert np.all(shape.data == data)
    assert shape.data_displayed.shape == (6, 2)
    assert shape.slice_key.shape == (2, 0)


def test_nD_path():
    """Test creating Shape with a random nD path."""
    # Test a single six vertex 3D path
    np.random.seed(0)
    data = 20 * np.random.random((6, 3))
    shape = Path(data)
    assert np.all(shape.data == data)
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
    assert np.all(shape.data == data)
    assert shape.data_displayed.shape == (2, 2)
    assert shape.slice_key.shape == (2, 0)


def test_nD_line():
    """Test creating Shape with a random nD line."""
    # Test a single two vertex 3D line
    np.random.seed(0)
    data = 20 * np.random.random((2, 3))
    shape = Line(data)
    assert np.all(shape.data == data)
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
    assert np.all(shape.data == data)
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
    assert np.all(shape.data == data)
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 1)

    shape.ndisplay = 3
    assert shape.data_displayed.shape == (4, 3)
