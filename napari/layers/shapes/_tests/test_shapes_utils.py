import numpy as np

from napari.layers.shapes._shapes_utils import (
    get_default_shape_type,
    number_of_shapes,
)


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
