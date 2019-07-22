import numpy as np
from xml.etree.ElementTree import Element
from vispy.color import Colormap
from napari.layers import Image


def test_random_image():
    """Test instantiating Image layer with random 2D data."""
    shape = (10, 15)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.range == tuple((0, m, 1) for m in shape)
    assert layer.multichannel == False
    assert layer._data_view.shape == shape[-2:]


def test_all_zeros_image():
    """Test instantiating Image layer with all zeros data."""
    shape = (10, 15)
    data = np.zeros(shape, dtype=float)
    layer = Image(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel == False
    assert layer._data_view.shape == shape[-2:]


def test_integer_image():
    """Test instantiating Image layer with integer data."""
    shape = (10, 15)
    np.random.seed(0)
    data = np.round(10 * np.random.random(shape)).astype(int)
    layer = Image(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel == False
    assert layer._data_view.shape == shape[-2:]


def test_bool_image():
    """Test instantiating Image layer with bool data."""
    shape = (10, 15)
    data = np.zeros(shape, dtype=bool)
    layer = Image(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel == False
    assert layer._data_view.shape == shape[-2:]


def test_3D_image():
    """Test instantiating Image layer with random 3D data."""
    shape = (10, 15, 6)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel == False
    assert layer._data_view.shape == shape[-2:]


def test_3D_image_shape_1():
    """Test instantiating Image layer with random 3D data with shape 1 axis."""
    shape = (1, 10, 15)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel == False
    assert layer._data_view.shape == shape[-2:]


def test_4D_image():
    """Test instantiating Image layer with random 4D data."""
    shape = (10, 15, 6, 8)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel == False
    assert layer._data_view.shape == shape[-2:]


def test_5D_image_shape_1():
    """Test instantiating Image layer with random 5D data with shape 1 axis."""
    shape = (4, 1, 2, 10, 15)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel == False
    assert layer._data_view.shape == shape[-2:]


def test_rgb_image():
    """Test instantiating Image layer with RGB data."""
    shape = (10, 15, 3)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape) - 1
    assert layer.shape == shape[:-1]
    assert layer.multichannel == True
    assert layer._data_view.shape == shape[-3:]


def test_rgba_image():
    """Test instantiating Image layer with RGBA data."""
    shape = (10, 15, 4)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape) - 1
    assert layer.shape == shape[:-1]
    assert layer.multichannel == True
    assert layer._data_view.shape == shape[-3:]


def test_non_rgb_image():
    """Test forcing Image layer to be 3D and not multichannel."""
    shape = (10, 15, 3)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data, multichannel=False)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel == False
    assert layer._data_view.shape == shape[-2:]


def test_non_multichannel_image():
    """Test forcing Image layer to be 3D and not multichannel."""
    # If multichannel is set to be True in constructor but the last dim has a
    # size > 4 then data cannot actually be multichannel
    shape = (10, 15, 6)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data, multichannel=True)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel == False
    assert layer._data_view.shape == shape[-2:]


def test_changing_image():
    """Test changing Image data."""
    shape_a = (10, 15)
    shape_b = (20, 12)
    np.random.seed(0)
    data_a = np.random.random(shape_a)
    data_b = np.random.random(shape_b)
    layer = Image(data_a)
    layer.data = data_b
    assert np.all(layer.data == data_b)
    assert layer.ndim == len(shape_b)
    assert layer.shape == shape_b
    assert layer.range == tuple((0, m, 1) for m in shape_b)
    assert layer.multichannel == False
    assert layer._data_view.shape == shape_b[-2:]


def test_changing_image_dims():
    """Test changing Image data including dimensionality."""
    shape_a = (10, 15)
    shape_b = (20, 12, 6)
    np.random.seed(0)
    data_a = np.random.random(shape_a)
    data_b = np.random.random(shape_b)
    layer = Image(data_a)

    # Prep indices for swtich to 3D
    layer._indices = (0,) + layer._indices
    layer.data = data_b
    assert np.all(layer.data == data_b)
    assert layer.ndim == len(shape_b)
    assert layer.shape == shape_b
    assert layer.range == tuple((0, m, 1) for m in shape_b)
    assert layer.multichannel == False
    assert layer._data_view.shape == shape_b[-2:]


def test_name():
    """Test setting layer name."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.name == 'Image'

    layer = Image(data, name='random')
    assert layer.name == 'random'

    layer.name = 'img'
    assert layer.name == 'img'


def test_visiblity():
    """Test setting layer visiblity."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.visible == True

    layer.visible = False
    assert layer.visible == False

    layer = Image(data, visible=False)
    assert layer.visible == False

    layer.visible = True
    assert layer.visible == True


def test_opacity():
    """Test setting layer opacity."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.opacity == 1

    layer.opacity = 0.5
    assert layer.opacity == 0.5

    layer = Image(data, opacity=0.6)
    assert layer.opacity == 0.6

    layer.opacity = 0.3
    assert layer.opacity == 0.3


def test_blending():
    """Test setting layer blending."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Image(data, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


def test_interpolation():
    """Test setting image interpolation mode."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.interpolation == 'nearest'

    layer = Image(data, interpolation='bicubic')
    assert layer.interpolation == 'bicubic'

    layer.interpolation = 'bilinear'
    assert layer.interpolation == 'bilinear'


def test_colormaps():
    """Test setting test_colormaps."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
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

    layer = Image(data, colormap='magma')
    assert layer.colormap[0] == 'magma'
    assert type(layer.colormap[1]) == Colormap

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.2, 1.0]])
    layer = Image(data, colormap=('custom', cmap))
    assert layer.colormap[0] == 'custom'
    assert layer.colormap[1] == cmap

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.6, 1.0]])
    layer = Image(data, colormap={'new': cmap})
    assert layer.colormap[0] == 'new'
    assert layer.colormap[1] == cmap


def test_clims():
    """Test setting color limits."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
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
    layer = Image(data, clim=clim)
    assert layer.clim == clim
    assert layer._clim_range == clim


def test_clim_range():
    """Test setting color limits range."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer._clim_range[0] >= 0
    assert layer._clim_range[1] <= 1
    assert layer._clim_range[0] < layer._clim_range[1]

    # If all data is the same value the clim_range and clim defaults to [0, 1]
    data = np.zeros((10, 15))
    layer = Image(data)
    assert layer._clim_range == [0, 1]
    assert layer.clim == [0.0, 1.0]

    # Set clim_range as keyword argument
    data = np.random.random((10, 15))
    layer = Image(data, clim_range=[0, 2])
    assert layer._clim_range == [0, 2]

    # Set clim and clim_range as keyword arguments
    data = np.random.random((10, 15))
    layer = Image(data, clim=[0.3, 0.6], clim_range=[0, 2])
    assert layer.clim == [0.3, 0.6]
    assert layer._clim_range == [0, 2]


def test_metadata():
    """Test setting image metadata."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.metadata == {}

    layer = Image(data, metadata={'unit': 'cm'})
    assert layer.metadata == {'unit': 'cm'}


def test_value():
    """Test getting the value of the data at the current coordinates."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    coord, value = layer.get_value()
    assert np.all(coord == [0, 0])
    assert value == data[0, 0]


def test_message():
    """Test converting value and coords to message."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    coord, value = layer.get_value()
    msg = layer.get_message(coord, value)
    assert type(msg) == str


def test_thumbnail():
    """Test the image thumbnail for square data."""
    np.random.seed(0)
    data = np.random.random((30, 30))
    layer = Image(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_xml_list():
    """Test the xml generation."""
    np.random.seed(0)
    data = np.random.random((15, 30))
    layer = Image(data)
    xml = layer.to_xml_list()
    assert type(xml) == list
    assert len(xml) == 1
    assert type(xml[0]) == Element
