from napari.layers import Image
import numpy as np


def test_random_image():
    """Test instantiating Image layer with random 2D data."""
    shape = (10, 15)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.all(layer.image == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.range == tuple((0, m, 1) for m in shape)
    assert layer.multichannel == False


def test_all_zeros_image():
    """Test instantiating Image layer with all zeros data."""
    shape = (10, 15)
    data = np.zeros(shape, dtype=float)
    layer = Image(data)
    assert np.all(layer.image == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel == False


def test_integer_image():
    """Test instantiating Image layer with integer data."""
    shape = (10, 15)
    data = np.round(10 * np.random.random(shape)).astype(int)
    layer = Image(data)
    assert np.all(layer.image == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel == False


def test_3D_image():
    """Test instantiating Image layer with random 3D data."""
    shape = (10, 15, 6)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.all(layer.image == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel == False


def test_4D_image():
    """Test instantiating Image layer with random 4D data."""
    shape = (10, 15, 6, 8)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.all(layer.image == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel == False


def test_rgb_image():
    """Test instantiating Image layer with RGB data."""
    shape = (10, 15, 3)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.all(layer.image == data)
    assert layer.ndim == len(shape) - 1
    assert layer.shape == shape[:-1]
    assert layer.multichannel == True


def test_rgba_image():
    """Test instantiating Image layer with RGBA data."""
    shape = (10, 15, 4)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.all(layer.image == data)
    assert layer.ndim == len(shape) - 1
    assert layer.shape == shape[:-1]
    assert layer.multichannel == True


def test_non_rgb_image():
    """Test forcing Image layer to be 3D and not multichannel."""
    shape = (10, 15, 3)
    data = np.random.random(shape)
    layer = Image(data, multichannel=False)
    assert np.all(layer.image == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel == False


# def test_non_multichannel_image():
#     """Test forcing Image layer to be 3D and not multichannel."""
#     # If multichannel is set to be True in constructor but the last dim has a
#     # size > 4 then data cannot actually be multichannel
#     shape = (10, 15, 6)
#     data = np.random.random(shape)
#     layer = Image(data, multichannel=True)
#     assert np.all(layer.image == data)
#     assert layer.ndim == len(shape)
#     assert layer.shape == shape
#     assert layer.multichannel == False


# def test_changing_image_data():
#     """Test changing Image data."""
#     shape_a = (10, 15)
#     shape_b = (20, 12)
#     data_a = np.random.random(shape_a)
#     data_b = np.random.random(shape_b)
#     layer = Image(data_a)
#     layer.image = data_b
#     assert np.all(layer.image == data_b)
#     assert layer.ndim == len(shape_b)
#     assert layer.shape == shape_b
#     assert layer.range == tuple((0, m, 1) for m in shape_b)
#     assert layer.multichannel == False


# def test_changing_image_dims():
#     """Test changing Image data."""
#     shape_a = (10, 15)
#     shape_b = (20, 12, 6)
#     data_a = np.random.random(shape_a)
#     data_b = np.random.random(shape_b)
#     layer = Image(data_a)
#     layer.image = data_b
#     assert np.all(layer.image == data_b)
#     assert layer.ndim == len(shape_b)
#     assert layer.shape == shape_b
#     assert layer.range == tuple((0, m, 1) for m in shape_b)
#     assert layer.multichannel == False


def test_name():
    """Test layer name."""
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.name == 'Image'

    layer = Image(data, name='random')
    assert layer.name == 'random'

    layer.name = 'img'
    assert layer.name == 'img'


def test_interpolation():
    """Test image interpolation mode."""
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.interpolation == 'nearest'

    layer = Image(data, interpolation='bicubic')
    assert layer.interpolation == 'bicubic'

    layer.interpolation = 'bilinear'
    assert layer.interpolation == 'bilinear'


#
