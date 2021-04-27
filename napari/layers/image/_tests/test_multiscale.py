import numpy as np
from skimage.transform import pyramid_gaussian

from napari._tests.utils import check_layer_world_data_extent
from napari.layers import Image
from napari.utils import Colormap


def test_random_multiscale():
    """Test instantiating Image layer with random 2D multiscale data."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    assert layer.data == data
    assert layer.multiscale is True
    assert layer.ndim == len(shapes[0])
    np.testing.assert_array_equal(layer.extent.data[1] + 1, shapes[0])
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_infer_multiscale():
    """Test instantiating Image layer with random 2D multiscale data."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data)
    assert layer.data == data
    assert layer.multiscale is True
    assert layer.ndim == len(shapes[0])
    np.testing.assert_array_equal(layer.extent.data[1] + 1, shapes[0])
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_infer_tuple_multiscale():
    """Test instantiating Image layer with random 2D multiscale data."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = tuple(np.random.random(s) for s in shapes)
    layer = Image(data)
    assert layer.data == data
    assert layer.multiscale is True
    assert layer.ndim == len(shapes[0])
    np.testing.assert_array_equal(layer.extent.data[1] + 1, shapes[0])
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_blocking_multiscale():
    """Test instantiating Image layer blocking 2D multiscale data."""
    shape = (40, 20)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data, multiscale=False)
    assert np.all(layer.data == data)
    assert layer.multiscale is False
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1] + 1, shape)
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_multiscale_tuple():
    """Test instantiating Image layer multiscale tuple."""
    shape = (40, 20)
    np.random.seed(0)
    img = np.random.random(shape)
    data = tuple(pyramid_gaussian(img, multichannel=False))
    layer = Image(data)
    assert layer.data == data
    assert layer.multiscale is True
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1] + 1, shape)
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_3D_multiscale():
    """Test instantiating Image layer with 3D data."""
    shapes = [(8, 40, 20), (4, 20, 10), (2, 10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    assert layer.data == data
    assert layer.ndim == len(shapes[0])
    np.testing.assert_array_equal(layer.extent.data[1] + 1, shapes[0])
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_non_uniform_3D_multiscale():
    """Test instantiating Image layer non-uniform 3D data."""
    shapes = [(8, 40, 20), (8, 20, 10), (8, 10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    assert layer.data == data
    assert layer.ndim == len(shapes[0])
    np.testing.assert_array_equal(layer.extent.data[1] + 1, shapes[0])
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_rgb_multiscale():
    """Test instantiating Image layer with RGB data."""
    shapes = [(40, 20, 3), (20, 10, 3), (10, 5, 3)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    assert layer.data == data
    assert layer.ndim == len(shapes[0]) - 1
    np.testing.assert_array_equal(layer.extent.data[1] + 1, shapes[0][:-1])
    assert layer.rgb is True
    assert layer._data_view.ndim == 3


def test_3D_rgb_multiscale():
    """Test instantiating Image layer with 3D RGB data."""
    shapes = [(8, 40, 20, 3), (4, 20, 10, 3), (2, 10, 5, 3)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    assert layer.data == data
    assert layer.ndim == len(shapes[0]) - 1
    np.testing.assert_array_equal(layer.extent.data[1] + 1, shapes[0][:-1])
    assert layer.rgb is True
    assert layer._data_view.ndim == 3


def test_non_rgb_image():
    """Test forcing Image layer to be 3D and not rgb."""
    shapes = [(40, 20, 3), (20, 10, 3), (10, 5, 3)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True, rgb=False)
    assert layer.data == data
    assert layer.ndim == len(shapes[0])
    np.testing.assert_array_equal(layer.extent.data[1] + 1, shapes[0])
    assert layer.rgb is False


def test_name():
    """Test setting layer name."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    assert layer.name == 'Image'

    layer = Image(data, multiscale=True, name='random')
    assert layer.name == 'random'

    layer.name = 'img'
    assert layer.name == 'img'


def test_visiblity():
    """Test setting layer visibility."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer = Image(data, multiscale=True, visible=False)
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_opacity():
    """Test setting layer opacity."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    assert layer.opacity == 1.0

    layer.opacity = 0.5
    assert layer.opacity == 0.5

    layer = Image(data, multiscale=True, opacity=0.6)
    assert layer.opacity == 0.6

    layer.opacity = 0.3
    assert layer.opacity == 0.3


def test_blending():
    """Test setting layer blending."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Image(data, multiscale=True, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


def test_interpolation():
    """Test setting image interpolation mode."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    assert layer.interpolation == 'nearest'

    layer = Image(data, multiscale=True, interpolation='bicubic')
    assert layer.interpolation == 'bicubic'

    layer.interpolation = 'bilinear'
    assert layer.interpolation == 'bilinear'


def test_colormaps():
    """Test setting test_colormaps."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    assert layer.colormap.name == 'gray'
    assert isinstance(layer.colormap, Colormap)

    layer.colormap = 'magma'
    assert layer.colormap.name == 'magma'
    assert isinstance(layer.colormap, Colormap)

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.2, 1.0]])
    layer.colormap = 'custom', cmap
    assert layer.colormap.name == 'custom'
    assert layer.colormap == cmap

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.6, 1.0]])
    layer.colormap = {'new': cmap}
    assert layer.colormap.name == 'new'
    assert layer.colormap == cmap

    layer = Image(data, multiscale=True, colormap='magma')
    assert layer.colormap.name == 'magma'
    assert isinstance(layer.colormap, Colormap)

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.2, 1.0]])
    layer = Image(data, multiscale=True, colormap=('custom', cmap))
    assert layer.colormap.name == 'custom'
    assert layer.colormap == cmap

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.6, 1.0]])
    layer = Image(data, multiscale=True, colormap={'new': cmap})
    assert layer.colormap.name == 'new'
    assert layer.colormap == cmap


def test_contrast_limits():
    """Test setting color limits."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
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
    layer = Image(data, multiscale=True, contrast_limits=contrast_limits)
    assert layer.contrast_limits == contrast_limits
    assert layer._contrast_limits_range == contrast_limits


def test_contrast_limits_range():
    """Test setting color limits range."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    assert layer._contrast_limits_range[0] >= 0
    assert layer._contrast_limits_range[1] <= 1
    assert layer._contrast_limits_range[0] < layer._contrast_limits_range[1]

    # If all data is the same value the contrast_limits_range and
    # contrast_limits defaults to [0, 1]
    shapes = [(40, 20), (20, 10), (10, 5)]
    data = [np.zeros(s) for s in shapes]
    layer = Image(data, multiscale=True)
    assert layer._contrast_limits_range == [0, 1]
    assert layer.contrast_limits == [0.0, 1.0]


def test_metadata():
    """Test setting image metadata."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    assert layer.metadata == {}

    layer = Image(data, multiscale=True, metadata={'unit': 'cm'})
    assert layer.metadata == {'unit': 'cm'}


def test_value():
    """Test getting the value of the data at the current coordinates."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    value = layer.get_value((0,) * 2)
    assert layer.data_level == 2
    np.testing.assert_allclose(value, (2, data[2][0, 0]))


def test_corner_value():
    """Test getting the value of the data at the new position."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    value = layer.get_value((0,) * 2)
    target_position = (39, 19)
    target_level = 0
    layer.data_level = target_level
    layer.corner_pixels[1] = shapes[target_level]  # update requested view
    layer.refresh()

    # Test position at corner of image
    value = layer.get_value(target_position)
    np.testing.assert_allclose(
        value, (target_level, data[target_level][target_position])
    )

    # Test position at outside image
    value = layer.get_value((40, 20))
    assert value[1] is None


def test_message():
    """Test converting value and coords to message."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    msg = layer.get_status((0,) * 2)
    assert type(msg) == str


def test_thumbnail():
    """Test the image thumbnail for square data."""
    shapes = [(40, 40), (20, 20), (10, 10)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_not_create_random_multiscale():
    """Test instantiating Image layer with random 2D data."""
    shape = (20_000, 20)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.all(layer.data == data)
    assert layer.multiscale is False


def test_world_data_extent():
    """Test extent after applying transforms."""
    np.random.seed(0)
    shapes = [(6, 40, 80), (3, 20, 40), (1, 10, 20)]
    data = [np.random.random(s) for s in shapes]
    layer = Image(data)
    extent = np.array(((0,) * 3, np.subtract(shapes[0], 1)))
    check_layer_world_data_extent(layer, extent, (3, 1, 1), (10, 20, 5))


def test_5D_multiscale():
    """Test 5D multiscale data."""
    shapes = [(1, 2, 5, 20, 20), (1, 2, 5, 10, 10), (1, 2, 5, 5, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = Image(data, multiscale=True)
    assert layer.data == data
    assert layer.multiscale is True
    assert layer.ndim == len(shapes[0])
