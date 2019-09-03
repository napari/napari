import numpy as np
from xml.etree.ElementTree import Element
from napari.layers import Surface


def test_random_surface():
    """Test instantiating Surface layer with random 2D data."""
    np.random.seed(0)
    vertices = np.random.random((10, 2))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)
    assert np.all(layer.data[0] == data[0])
    assert np.all(layer.data[1] == data[1])
    assert np.all(layer.data[2] == data[2])
