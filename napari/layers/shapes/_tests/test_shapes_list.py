import numpy as np
from napari.layers.shapes._shapes_list import ShapesList
from napari.layers.shapes._shapes_models import Rectangle, Polygon, Path


def test_empty_shape_list():
    """Test instantiating empty ShapesList."""
    shape_list = ShapesList()
    assert len(shape_list.shapes) == 0


def test_adding_to_shape_list():
    """Test adding shapes to ShapesList."""
    np.random.seed(0)
    data = 20 * np.random.random((4, 2))
    shape = Rectangle(data)
    shape_list = ShapesList()

    shape_list.add(shape)
    assert len(shape_list.shapes) == 1
    assert shape_list.shapes[0] == shape


def test_nD_shapes():
    """Test adding shapes to ShapesList."""
    np.random.seed(0)
    shape_list = ShapesList()
    data = 20 * np.random.random((6, 3))
    data[:, 0] = 0
    shape_a = Polygon(data)
    shape_list.add(shape_a)

    data = 20 * np.random.random((6, 3))
    data[:, 0] = 1
    shape_b = Path(data)
    shape_list.add(shape_b)

    assert len(shape_list.shapes) == 2
    assert shape_list.shapes[0] == shape_a
    assert shape_list.shapes[1] == shape_b

    assert shape_list._vertices.shape[1] == 2
    assert shape_list._mesh.vertices.shape[1] == 2

    shape_list.ndisplay = 3
    assert shape_list._vertices.shape[1] == 3
    assert shape_list._mesh.vertices.shape[1] == 3
