import numpy as np
import pytest

from napari.layers.shapes._shape_list import ShapeList
from napari.layers.shapes._shapes_models import Path, Polygon, Rectangle


def test_empty_shape_list():
    """Test instantiating empty ShapeList."""
    shape_list = ShapeList()
    assert len(shape_list.shapes) == 0


def test_adding_to_shape_list():
    """Test adding shapes to ShapeList."""
    np.random.seed(0)
    data = 20 * np.random.random((4, 2))
    shape = Rectangle(data)
    shape_list = ShapeList()

    shape_list.add(shape)
    assert len(shape_list.shapes) == 1
    assert shape_list.shapes[0] == shape


def test_nD_shapes():
    """Test adding shapes to ShapeList."""
    np.random.seed(0)
    shape_list = ShapeList()
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


@pytest.mark.parametrize("attribute", ['edge', 'face'])
def test_bad_color_array(attribute):
    """Test adding shapes to ShapeList."""
    np.random.seed(0)
    data = 20 * np.random.random((4, 2))
    shape = Rectangle(data)
    shape_list = ShapeList()

    shape_list.add(shape)

    # test setting color with a color array of the wrong shape
    bad_color_array = np.array([[0, 0, 0, 1], [1, 1, 1, 1]])
    with pytest.raises(ValueError):
        setattr(shape_list, f'{attribute}_color', bad_color_array)
