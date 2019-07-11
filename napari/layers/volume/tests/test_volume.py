import numpy as np
from vispy.color import Colormap
from napari.util.colormaps.colormaps import TransGrays, TransFire
from napari.layers import Volume


def test_random_volume():
    """Test instantiating Volume layer with random 2D data."""
    shape = (10, 15, 20)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Volume(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.range == tuple((0, m, 1) for m in shape)
    assert layer.multichannel is False
    assert layer._data_view.shape == shape[-3:]


def test_all_zeros_volume():
    """Test instantiating Volume layer with all zeros data."""
    shape = (10, 15, 20)
    data = np.zeros(shape, dtype=float)
    layer = Volume(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel is False
    assert layer._data_view.shape == shape[-3:]


def test_integer_volume():
    """Test instantiating Volume layer with integer data."""
    shape = (10, 15, 20)
    np.random.seed(0)
    data = np.round(10 * np.random.random(shape)).astype(int)
    layer = Volume(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel is False
    assert layer._data_view.shape == shape[-3:]


def test_3D_volume():
    """Test instantiating Volume layer with random 3D data."""
    shape = (10, 15, 6)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Volume(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel is False
    assert layer._data_view.shape == shape[-3:]


def test_rgb_volume():
    """Test instantiating Volume layer with RGB data."""
    shape = (10, 15, 20, 3)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Volume(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape) - 1
    assert layer.shape == shape[:-1]
    assert layer.multichannel is True
    assert layer._data_view.shape == shape[-4:]


def test_rgba_volume():
    """Test instantiating Volume layer with RGBA data."""
    shape = (10, 15, 20, 4)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Volume(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape) - 1
    assert layer.shape == shape[:-1]
    assert layer.multichannel is True
    assert layer._data_view.shape == shape[-4:]


def test_non_rgb_volume():
    """Test forcing Volume layer to be 3D and not multichannel."""
    shape = (10, 15, 20)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Volume(data, multichannel=False)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.multichannel is False
    assert layer._data_view.shape == shape[-3:]


def test_changing_volume():
    """Test changing Volume data."""
    shape_a = (10, 15, 30)
    shape_b = (20, 12, 4)
    np.random.seed(0)
    data_a = np.random.random(shape_a)
    data_b = np.random.random(shape_b)
    layer = Volume(data_a)
    layer.data = data_b
    assert np.all(layer.data == data_b)
    assert layer.ndim == len(shape_b)
    assert layer.shape == shape_b
    assert layer.range == tuple((0, m, 1) for m in shape_b)
    assert layer.multichannel is False
    assert layer._data_view.shape == shape_b[-3:]


def test_name():
    """Test setting layer name."""
    np.random.seed(0)
    data = np.random.random((10, 15, 40))
    layer = Volume(data)
    assert layer.name == 'Volume'

    layer = Volume(data, name='random')
    assert layer.name == 'random'

    layer.name = 'img'
    assert layer.name == 'img'


def test_colormaps():
    """Test setting test_colormaps."""
    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    layer = Volume(data)
    assert layer.colormap[0] == 'grays'
    assert type(layer.colormap[1]) == TransGrays

    layer.colormap = 'fire'
    assert layer.colormap[0] == 'fire'
    assert type(layer.colormap[1]) == TransFire

    layer.colormap = 'gray'
    assert layer.colormap[0] == 'gray'
    assert type(layer.colormap[1]) == Colormap


def test_metadata():
    """Test setting image metadata."""
    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    layer = Volume(data)
    assert layer.metadata == {}

    layer = Volume(data, metadata={'unit': 'cm'})
    assert layer.metadata == {'unit': 'cm'}
