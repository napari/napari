import numpy as np
from skimage.transform import pyramid_gaussian
from xml.etree.ElementTree import Element
from vispy.color import Colormap
from napari.layers import Image
import pytest


def test_random_pyramid():
    """Test instantiating Image layer with random 2D pyramid data."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    assert layer.data == data
    assert layer.is_pyramid is True
    assert len(layer._data_pyramid) > 0
    assert layer.ndim == len(shapes[0])
    assert layer.shape == shapes[0]
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_infer_pyramid():
    """Test instantiating Image layer with random 2D pyramid data."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data)
    assert layer.data == data
    assert layer.is_pyramid is True
    assert len(layer._data_pyramid) > 0
    assert layer.ndim == len(shapes[0])
    assert layer.shape == shapes[0]
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_error_pyramid():
    """Test error on forcing non pyramid."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    with pytest.raises(ValueError):
        Image(data, is_pyramid=False)


def test_infer_tuple_pyramid():
    """Test instantiating Image layer with random 2D pyramid data."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = tuple(np.random.random(s) for s in shapes)
    layer = Image(data)
    assert layer.data == data
    assert layer.is_pyramid is True
    assert layer.ndim == len(shapes[0])
    assert layer.shape == shapes[0]
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_forcing_pyramid():
    """Test instantiating Image layer forcing 2D pyramid data."""
    shape = (40, 20)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data, is_pyramid=True)
    assert np.all(layer.data == data)
    assert layer.is_pyramid is True
    assert len(layer._data_pyramid) > 0
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_blocking_pyramid():
    """Test instantiating Image layer blocking 2D pyramid data."""
    shape = (40, 20)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data, is_pyramid=False)
    assert np.all(layer.data == data)
    assert layer.is_pyramid is False
    assert layer._data_pyramid is None
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_pyramid_tuple():
    """Test instantiating Image layer pyramid tuple."""
    shape = (40, 20)
    np.random.seed(0)
    img = np.random.random(shape)
    data = tuple(pyramid_gaussian(img, multichannel=False))
    layer = Image(data)
    assert layer.data == data
    assert layer.is_pyramid is True
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_3D_pyramid():
    """Test instantiating Image layer with 3D data."""
    shapes = [(8, 40, 20), (4, 20, 10), (2, 10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    assert layer.data == data
    assert layer.ndim == len(shapes[0])
    assert layer.shape == shapes[0]
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_non_uniform_3D_pyramid():
    """Test instantiating Image layer non-uniform 3D data."""
    shapes = [(8, 40, 20), (8, 20, 10), (8, 10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    assert layer.data == data
    assert layer.ndim == len(shapes[0])
    assert layer.shape == shapes[0]
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_rgb_pyramid():
    """Test instantiating Image layer with RGB data."""
    shapes = [(40, 20, 3), (20, 10, 3), (10, 5, 3)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    assert layer.data == data
    assert layer.ndim == len(shapes[0]) - 1
    assert layer.shape == shapes[0][:-1]
    assert layer.rgb is True
    assert layer._data_view.ndim == 3


def test_3D_rgb_pyramid():
    """Test instantiating Image layer with 3D RGB data."""
    shapes = [(8, 40, 20, 3), (4, 20, 10, 3), (2, 10, 5, 3)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    assert layer.data == data
    assert layer.ndim == len(shapes[0]) - 1
    assert layer.shape == shapes[0][:-1]
    assert layer.rgb is True
    assert layer._data_view.ndim == 3


def test_non_rgb_image():
    """Test forcing Image layer to be 3D and not rgb."""
    shapes = [(40, 20, 3), (20, 10, 3), (10, 5, 3)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True, rgb=False)
    assert layer.data == data
    assert layer.ndim == len(shapes[0])
    assert layer.shape == shapes[0]
    assert layer.rgb is False


def test_name():
    """Test setting layer name."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    assert layer.name == 'Image'

    layer = Image(data, is_pyramid=True, name='random')
    assert layer.name == 'random'

    layer.name = 'img'
    assert layer.name == 'img'


def test_visiblity():
    """Test setting layer visiblity."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer = Image(data, is_pyramid=True, visible=False)
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_opacity():
    """Test setting layer opacity."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    assert layer.opacity == 1.0

    layer.opacity = 0.5
    assert layer.opacity == 0.5

    layer = Image(data, is_pyramid=True, opacity=0.6)
    assert layer.opacity == 0.6

    layer.opacity = 0.3
    assert layer.opacity == 0.3


def test_blending():
    """Test setting layer blending."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Image(data, is_pyramid=True, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


def test_interpolation():
    """Test setting image interpolation mode."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    assert layer.interpolation == 'nearest'

    layer = Image(data, is_pyramid=True, interpolation='bicubic')
    assert layer.interpolation == 'bicubic'

    layer.interpolation = 'bilinear'
    assert layer.interpolation == 'bilinear'


def test_colormaps():
    """Test setting test_colormaps."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
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

    layer = Image(data, is_pyramid=True, colormap='magma')
    assert layer.colormap[0] == 'magma'
    assert type(layer.colormap[1]) == Colormap

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.2, 1.0]])
    layer = Image(data, is_pyramid=True, colormap=('custom', cmap))
    assert layer.colormap[0] == 'custom'
    assert layer.colormap[1] == cmap

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.6, 1.0]])
    layer = Image(data, is_pyramid=True, colormap={'new': cmap})
    assert layer.colormap[0] == 'new'
    assert layer.colormap[1] == cmap


def test_contrast_limits():
    """Test setting color limits."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
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
    layer = Image(data, is_pyramid=True, contrast_limits=contrast_limits)
    assert layer.contrast_limits == contrast_limits
    assert layer._contrast_limits_range == contrast_limits


def test_contrast_limits_range():
    """Test setting color limits range."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    assert layer._contrast_limits_range[0] >= 0
    assert layer._contrast_limits_range[1] <= 1
    assert layer._contrast_limits_range[0] < layer._contrast_limits_range[1]

    # If all data is the same value the contrast_limits_range and
    # contrast_limits defaults to [0, 1]
    shapes = [(40, 20), (20, 10), (10, 5)]
    data = [np.zeros(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    assert layer._contrast_limits_range == [0, 1]
    assert layer.contrast_limits == [0.0, 1.0]


def test_metadata():
    """Test setting image metadata."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    assert layer.metadata == {}

    layer = Image(data, is_pyramid=True, metadata={'unit': 'cm'})
    assert layer.metadata == {'unit': 'cm'}


def test_value():
    """Test getting the value of the data at the current coordinates."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    value = layer.get_value()
    assert layer.coordinates == (0, 0)
    # Note that here, because the shapes of the pyramid are all very small
    # data that will be rendered will only ever come from the bottom two
    # levels of the pyramid.
    assert value == (1, data[1][0, 0])


def test_message():
    """Test converting value and coords to message."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    msg = layer.get_message()
    assert type(msg) == str


def test_thumbnail():
    """Test the image thumbnail for square data."""
    shapes = [(40, 40), (20, 20), (10, 10)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_xml_list():
    """Test the xml generation."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, is_pyramid=True)
    xml = layer.to_xml_list()
    assert type(xml) == list
    assert len(xml) == 1
    assert type(xml[0]) == Element


def test_create_random_pyramid():
    """Test instantiating Image layer with random 2D data."""
    shape = (20_000, 20)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.all(layer.data == data)
    assert layer.is_pyramid is True
    assert layer._data_pyramid[0].shape == shape
    assert layer._data_pyramid[1].shape == (shape[0] / 2, shape[1])
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.rgb is False
    assert layer._data_view.ndim == 2
