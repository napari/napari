import numpy as np
from xml.etree.ElementTree import Element
from vispy.color import Colormap
from napari.layers import Pyramid


# Set random seed for testing
np.random.seed(0)


def test_random_pyramid():
    """Test instantiating Pyramid layer with random 2D data."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer.data == data
    assert layer.ndim == len(shapes[0])
    assert layer.shape == shapes[0]
    assert layer.multichannel == False
    assert layer._data_view.ndim == 2


def test_3D_pyramid():
    """Test instantiating Pyramid layer with 3D data."""
    shapes = [(8, 40, 20), (4, 20, 10), (2, 10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer.data == data
    assert layer.ndim == len(shapes[0])
    assert layer.shape == shapes[0]
    assert layer.multichannel == False
    assert layer._data_view.ndim == 2


def test_non_uniform_3D_pyramid():
    """Test instantiating Pyramid layer non-uniform 3D data."""
    shapes = [(8, 40, 20), (8, 20, 10), (8, 10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer.data == data
    assert layer.ndim == len(shapes[0])
    assert layer.shape == shapes[0]
    assert layer.multichannel == False
    assert layer._data_view.ndim == 2


def test_rgb_pyramid():
    """Test instantiating Pyramid layer with RGB data."""
    shapes = [(40, 20, 3), (20, 10, 3), (10, 5, 3)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer.data == data
    assert layer.ndim == len(shapes[0]) - 1
    assert layer.shape == shapes[0][:-1]
    assert layer.multichannel == True
    assert layer._data_view.ndim == 3


def test_3D_rgb_pyramid():
    """Test instantiating Pyramid layer with 3D RGB data."""
    shapes = [(8, 40, 20, 3), (4, 20, 10, 3), (2, 10, 5, 3)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer.data == data
    assert layer.ndim == len(shapes[0]) - 1
    assert layer.shape == shapes[0][:-1]
    assert layer.multichannel == True
    assert layer._data_view.ndim == 3


def test_non_rgb_image():
    """Test forcing Pyramid layer to be 3D and not multichannel."""
    shapes = [(40, 20, 3), (20, 10, 3), (10, 5, 3)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data, multichannel=False)
    assert layer.data == data
    assert layer.ndim == len(shapes[0])
    assert layer.shape == shapes[0]
    assert layer.multichannel == False


def test_name():
    """Test setting layer name."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer.name == 'Pyramid'

    layer = Pyramid(data, name='random')
    assert layer.name == 'random'

    layer.name = 'img'
    assert layer.name == 'img'


def test_interpolation():
    """Test setting image interpolation mode."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer.interpolation == 'nearest'

    layer = Pyramid(data, interpolation='bicubic')
    assert layer.interpolation == 'bicubic'

    layer.interpolation = 'bilinear'
    assert layer.interpolation == 'bilinear'


def test_colormaps():
    """Test setting test_colormaps."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer.colormap[0] == 'gray'
    assert type(layer.colormap[1]) == Colormap

    layer.colormap = 'magma'
    assert layer.colormap[0] == 'magma'
    assert type(layer.colormap[1]) == Colormap

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.2, 1.0]])
    layer.colormap = 'custom', cmap
    assert layer.colormap[0] == 'custom'
    assert layer.colormap[1] == cmap

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.6, 1.0]])
    layer.colormap = {'new': cmap}
    assert layer.colormap[0] == 'new'
    assert layer.colormap[1] == cmap

    layer = Pyramid(data, colormap='magma')
    assert layer.colormap[0] == 'magma'
    assert type(layer.colormap[1]) == Colormap

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.2, 1.0]])
    layer = Pyramid(data, colormap=('custom', cmap))
    assert layer.colormap[0] == 'custom'
    assert layer.colormap[1] == cmap

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.6, 1.0]])
    layer = Pyramid(data, colormap={'new': cmap})
    assert layer.colormap[0] == 'new'
    assert layer.colormap[1] == cmap


def test_clims():
    """Test setting color limits."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer.clim[0] >= 0
    assert layer.clim[1] <= 1
    assert layer.clim[0] < layer.clim[1]
    assert layer.clim == layer._clim_range

    # Change clim property
    clim = [0, 2]
    layer.clim = clim
    assert layer.clim == clim
    assert layer._clim_range == clim

    # Set clim as keyword argument
    layer = Pyramid(data, clim=clim)
    assert layer.clim == clim
    assert layer._clim_range == clim


def test_clim_range():
    """Test setting color limits range."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer._clim_range[0] >= 0
    assert layer._clim_range[1] <= 1
    assert layer._clim_range[0] < layer._clim_range[1]

    # If all data is the same value the clim_range and clim defaults to [0, 1]
    shapes = [(40, 20), (20, 10), (10, 5)]
    data = [np.zeros(s) for s in shapes]
    layer = Pyramid(data)
    assert layer._clim_range == [0, 1]
    assert layer.clim == [0.0, 1.0]

    # Set clim_range as keyword argument
    shapes = [(40, 20), (20, 10), (10, 5)]
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data, clim_range=[0, 2])
    assert layer._clim_range == [0, 2]

    # Set clim and clim_range as keyword arguments
    shapes = [(40, 20), (20, 10), (10, 5)]
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data, clim=[0.3, 0.6], clim_range=[0, 2])
    assert layer.clim == [0.3, 0.6]
    assert layer._clim_range == [0, 2]


def test_metadata():
    """Test setting image metadata."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer.metadata == {}

    layer = Pyramid(data, metadata={'unit': 'cm'})
    assert layer.metadata == {'unit': 'cm'}


def test_value():
    """Test getting the value of the data at the current coordinates."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    coord, value = layer.get_value()
    assert np.all(coord == [0, 0])
    assert value == data[-1][0, 0]


def test_message():
    """Test converting value and coords to message."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    coord, value = layer.get_value()
    msg = layer.get_message(coord, value)
    assert type(msg) == str


def test_thumbnail():
    """Test the image thumbnail for square data."""
    shapes = [(40, 40), (20, 20), (10, 10)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_xml_list():
    """Test the xml generation."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    xml = layer.to_xml_list()
    assert type(xml) == list
    assert len(xml) == 1
    assert type(xml[0]) == Element
