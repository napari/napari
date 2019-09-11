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
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert np.all(layer.vertices == vertices)
    assert np.all(layer.faces == faces)
    assert np.all(layer.vertex_values == values)
    assert layer._data_view.shape[1] == 2


def test_random_3D_surface():
    """Test instantiating Surface layer with random 3D data."""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer._data_view.shape[1] == 2

    layer.dims.ndisplay = 3
    assert layer._data_view.shape[1] == 3
