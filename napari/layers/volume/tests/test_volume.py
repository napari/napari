import numpy as np
from napari.layers import Volume
from vispy.color import Colormap
from napari.util.colormaps.colormaps import TransFire


def test_random_volume():
    """Test instantiating Volume layer with random 3D data."""
    shape = (10, 15, 20)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Volume(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.range == tuple((0, m, 1) for m in shape)
    assert layer._data_view.shape == shape[-3:]


def test_all_zeros_volume():
    """Test instantiating Volume layer with all zeros data."""
    shape = (10, 15, 20)
    data = np.zeros(shape, dtype=float)
    layer = Volume(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
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
    assert layer._data_view.shape == shape[-3:]


def test_4D_volume():
    """Test instantiating multiple Volume layers with random 4D data."""
    shape = (10, 15, 6, 8)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Volume(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
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


def test_spacing():
    """Test instantiating anisotropic 3D volume."""
    shape = (10, 15, 20)
    spacing = [3, 1, 1]
    full_shape = tuple(np.multiply(shape, spacing))
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Volume(data, spacing=spacing)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == full_shape
    assert layer.range == tuple((0, m, 1) for m in full_shape)
    assert layer._data_view.shape == shape[-3:]


def test_visiblity():
    """Test setting layer visiblity."""
    np.random.seed(0)
    data = np.random.random((10, 15, 40))
    layer = Volume(data)
    assert layer.visible == True

    layer.visible = False
    assert layer.visible == False

    layer = Volume(data, visible=False)
    assert layer.visible == False

    layer.visible = True
    assert layer.visible == True


def test_opacity():
    """Test setting layer opacity."""
    np.random.seed(0)
    data = np.random.random((10, 15, 40))
    layer = Volume(data)
    assert layer.opacity == 1.0

    layer.opacity = 0.5
    assert layer.opacity == 0.5

    layer = Volume(data, opacity=0.6)
    assert layer.opacity == 0.6

    layer.opacity = 0.3
    assert layer.opacity == 0.3


def test_blending():
    """Test setting layer blending."""
    np.random.seed(0)
    data = np.random.random((10, 15, 40))
    layer = Volume(data)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Volume(data, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


def test_colormaps():
    """Test setting test_colormaps."""
    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    layer = Volume(data)
    assert layer.colormap[0] == 'gray'
    assert type(layer.colormap[1]) == Colormap

    layer.colormap = 'fire'
    assert layer.colormap[0] == 'fire'
    assert type(layer.colormap[1]) == TransFire

    layer.colormap = 'red'
    assert layer.colormap[0] == 'red'
    assert type(layer.colormap[1]) == Colormap


def test_metadata():
    """Test setting Volume metadata."""
    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    layer = Volume(data)
    assert layer.metadata == {}

    layer = Volume(data, metadata={'unit': 'cm'})
    assert layer.metadata == {'unit': 'cm'}


def test_clims():
    """Test setting color limits."""
    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    layer = Volume(data)
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
    layer = Volume(data, clim=clim)
    assert layer.clim == clim
    assert layer._clim_range == clim


def test_clim_range():
    """Test setting color limits range."""
    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    layer = Volume(data)
    assert layer._clim_range[0] >= 0
    assert layer._clim_range[1] <= 1
    assert layer._clim_range[0] < layer._clim_range[1]

    # If all data is the same value the clim_range and clim defaults to [0, 1]
    data = np.zeros((10, 15, 20))
    layer = Volume(data)
    assert layer._clim_range == [0, 1]
    assert layer.clim == [0.0, 1.0]

    # Set clim_range as keyword argument
    data = np.random.random((10, 15, 20))
    layer = Volume(data, clim_range=[0, 2])
    assert layer._clim_range == [0, 2]

    # Set clim and clim_range as keyword arguments
    data = np.random.random((10, 15, 20))
    layer = Volume(data, clim=[0.3, 0.6], clim_range=[0, 2])
    assert layer.clim == [0.3, 0.6]
    assert layer._clim_range == [0, 2]
