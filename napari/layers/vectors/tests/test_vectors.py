import numpy as np
from xml.etree.ElementTree import Element
from napari.layers import Vectors


# Set random seed for testing
np.random.seed(0)


def test_random_vectors():
    """Test instantiating Vectors layer with random coordinate-like 2D data."""
    shape = (10, 2, 2)
    np.random.seed(0)
    data = np.random.random(shape)
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert np.all(layer.data == data)
    assert layer.data.shape == shape
    assert layer.ndim == shape[2]
    assert layer._data_view.shape[2] == 2


def test_random_vectors_image():
    """Test instantiating Vectors layer with random image-like 2D data."""
    shape = (20, 10, 2)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Vectors(data)
    assert layer.data.shape == (20 * 10, 2, 2)
    assert layer.ndim == 2
    assert layer._data_view.shape[2] == 2


def test_empty_vectors():
    """Test instantiating Vectors layer with empty coordinate-like 2D data."""
    shape = (0, 2, 2)
    data = np.empty(shape)
    layer = Vectors(data)
    assert np.all(layer.data == data)
    assert layer.data.shape == shape
    assert layer.ndim == shape[2]
    assert layer._data_view.shape[2] == 2


def test_random_3D_vectors():
    """Test instantiating Vectors layer with random coordinate-like 3D data."""
    shape = (10, 2, 3)
    np.random.seed(0)
    data = np.random.random(shape)
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert np.all(layer.data == data)
    assert layer.data.shape == shape
    assert layer.ndim == shape[2]
    assert layer._data_view.shape[2] == 2


def test_random_3D_vectors_image():
    """Test instantiating Vectors layer with random image-like 3D data."""
    shape = (12, 20, 10, 3)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Vectors(data)
    assert layer.data.shape == (12 * 20 * 10, 2, 3)
    assert layer.ndim == 3
    assert layer._data_view.shape[2] == 2


def test_empty_3D_vectors():
    """Test instantiating Vectors layer with empty coordinate-like 3D data."""
    shape = (0, 2, 3)
    data = np.empty(shape)
    layer = Vectors(data)
    assert np.all(layer.data == data)
    assert layer.data.shape == shape
    assert layer.ndim == shape[2]
    assert layer._data_view.shape[2] == 2


def test_changing_data():
    """Test changing Vectors data."""
    shape_a = (10, 2, 2)
    np.random.seed(0)
    data_a = np.random.random(shape_a)
    data_a[:, 0, :] = 20 * data_a[:, 0, :]
    shape_b = (16, 2, 2)
    data_b = np.random.random(shape_b)
    data_b[:, 0, :] = 20 * data_b[:, 0, :]
    layer = Vectors(data_b)
    layer.data = data_b
    assert np.all(layer.data == data_b)
    assert layer.data.shape == shape_b
    assert layer.ndim == shape_b[2]
    assert layer._data_view.shape[2] == 2


def test_name():
    """Test setting layer name."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert layer.name == 'Vectors'

    layer = Vectors(data, name='random')
    assert layer.name == 'random'

    layer.name = 'vcts'
    assert layer.name == 'vcts'


def test_visiblity():
    """Test setting layer visiblity."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer = Vectors(data, visible=False)
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_opacity():
    """Test setting layer opacity."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert layer.opacity == 0.7

    layer.opacity = 0.5
    assert layer.opacity == 0.5

    layer = Vectors(data, opacity=0.6)
    assert layer.opacity == 0.6

    layer.opacity = 0.3
    assert layer.opacity == 0.3


def test_blending():
    """Test setting layer blending."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Vectors(data, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


def test_edge_width():
    """Test setting edge width."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert layer.edge_width == 1

    layer.edge_width = 2
    assert layer.edge_width == 2

    layer = Vectors(data, edge_width=3)
    assert layer.edge_width == 3


def test_edge_color():
    """Test setting edge color."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert layer.edge_color == 'red'

    layer.edge_color = 'blue'
    assert layer.edge_color == 'blue'

    layer = Vectors(data, edge_color='green')
    assert layer.edge_color == 'green'


def test_length():
    """Test setting length."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert layer.length == 1

    layer.length = 2
    assert layer.length == 2

    layer = Vectors(data, length=3)
    assert layer.length == 3


def test_thumbnail():
    """Test the image thumbnail for square data."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 18 * data[:, 0, :] + 1
    data[0, :, :] = [0, 0]
    data[-1, 0, :] = [20, 20]
    data[-1, 1, :] = [0, 0]
    layer = Vectors(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_value():
    """Test getting the value of the data at the current coordinates."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    value = layer.get_value()
    assert layer.coordinates == (0, 0)
    assert value is None


def test_message():
    """Test converting value and coords to message."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    msg = layer.get_message()
    assert type(msg) == str


def test_xml_list():
    """Test the xml generation."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    xml = layer.to_xml_list()
    assert type(xml) == list
    assert len(xml) == 10
    assert np.all([type(x) == Element for x in xml])
