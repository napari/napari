import numpy as np
from xml.etree.ElementTree import Element
from vispy.color import Colormap
from napari.layers import Pyramid


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


def test_visiblity():
    """Test setting layer visiblity."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer.visible == True

    layer.visible = False
    assert layer.visible == False

    layer = Pyramid(data, visible=False)
    assert layer.visible == False

    layer.visible = True
    assert layer.visible == True


def test_opacity():
    """Test setting layer opacity."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer.opacity == 1.0

    layer.opacity = 0.5
    assert layer.opacity == 0.5

    layer = Pyramid(data, opacity=0.6)
    assert layer.opacity == 0.6

    layer.opacity = 0.3
    assert layer.opacity == 0.3


def test_blending():
    """Test setting layer blending."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Pyramid(data, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


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


def test_contrast_limits():
    """Test setting color limits."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer.contrast_limits[0] >= 0
    assert layer.contrast_limits[1] <= 1
    assert layer.contrast_limits[0] < layer.contrast_limits[1]
    assert layer.contrast_limits == layer._contrast_limits_range

    # Change contrast_limits property
    contrast_limits = [0, 2]
    layer.contrast_limits = contrast_limits
    assert layer.contrast_limits == contrast_limits
    assert layer._contrast_limits_range == contrast_limits

    # Set contrast_limits as keyword argument
    layer = Pyramid(data, contrast_limits=contrast_limits)
    assert layer.contrast_limits == contrast_limits
    assert layer._contrast_limits_range == contrast_limits


def test_contrast_limits_range():
    """Test setting color limits range."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    assert layer._contrast_limits_range[0] >= 0
    assert layer._contrast_limits_range[1] <= 1
    assert layer._contrast_limits_range[0] < layer._contrast_limits_range[1]

    # If all data is the same value the contrast_limits_range and contrast_limits defaults to [0, 1]
    shapes = [(40, 20), (20, 10), (10, 5)]
    data = [np.zeros(s) for s in shapes]
    layer = Pyramid(data)
    assert layer._contrast_limits_range == [0, 1]
    assert layer.contrast_limits == [0.0, 1.0]

    # Set contrast_limits_range as keyword argument
    shapes = [(40, 20), (20, 10), (10, 5)]
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data, contrast_limits_range=[0, 2])
    assert layer._contrast_limits_range == [0, 2]

    # Set contrast_limits and contrast_limits_range as keyword arguments
    shapes = [(40, 20), (20, 10), (10, 5)]
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(
        data, contrast_limits=[0.3, 0.6], contrast_limits_range=[0, 2]
    )
    assert layer.contrast_limits == [0.3, 0.6]
    assert layer._contrast_limits_range == [0, 2]


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
    value = layer.get_value()
    assert layer.coordinates == (0, 0)
    assert value == (2, data[-1][0, 0])


def test_message():
    """Test converting value and coords to message."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Pyramid(data)
    msg = layer.get_message()
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
