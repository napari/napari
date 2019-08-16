import numpy as np
from copy import copy
from napari.layers.shapes.shape_list import ShapeList
from napari.layers.shapes.shape_models import Rectangle


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
