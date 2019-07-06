import numpy as np
from xml.etree.ElementTree import Element
from napari.layers import Labels


def test_random_labels():
    """Test instantiating Labels layer with random 2D data."""
    shape = (10, 15)
    data = np.round(20 * np.random.random(shape)).astype(int)
    layer = Labels(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.range == tuple((0, m, 1) for m in shape)
    assert layer._data_view.shape == shape[-2:]


def test_all_zeros_labels():
    """Test instantiating Labels layer with all zeros data."""
    shape = (10, 15)
    data = np.zeros(shape, dtype=int)
    layer = Labels(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer._data_view.shape == shape[-2:]


def test_3D_labels():
    """Test instantiating Labels layer with random 3D data."""
    shape = (6, 10, 15)
    data = np.round(20 * np.random.random(shape)).astype(int)
    layer = Labels(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer._data_view.shape == shape[-2:]


def test_changing_labels():
    """Test changing Labels data."""
    shape_a = (10, 15)
    shape_b = (20, 12)
    data_a = np.round(20 * np.random.random(shape_a)).astype(int)
    data_b = np.round(20 * np.random.random(shape_b)).astype(int)
    layer = Labels(data_a)
    layer.data = data_b
    assert np.all(layer.data == data_b)
    assert layer.ndim == len(shape_b)
    assert layer.shape == shape_b
    assert layer.range == tuple((0, m, 1) for m in shape_b)
    assert layer._data_view.shape == shape_b[-2:]


def test_changing_labels_dims():
    """Test changing Labels data including dimensionality."""
    shape_a = (10, 15)
    shape_b = (20, 12, 6)
    data_a = np.round(20 * np.random.random(shape_a)).astype(int)
    data_b = np.round(20 * np.random.random(shape_b)).astype(int)
    layer = Labels(data_a)

    # Prep indices for swtich to 3D
    layer._indices = (0,) + layer._indices
    layer.data = data_b
    assert np.all(layer.data == data_b)
    assert layer.ndim == len(shape_b)
    assert layer.shape == shape_b
    assert layer.range == tuple((0, m, 1) for m in shape_b)
    assert layer._data_view.shape == shape_b[-2:]


def test_name():
    """Test setting layer name."""
    data = np.round(20 * np.random.random((10, 15))).astype(int)
    layer = Labels(data)
    assert layer.name == 'Labels'

    layer = Labels(data, name='random')
    assert layer.name == 'random'

    layer.name = 'lbls'
    assert layer.name == 'lbls'


#
#
# def test_interpolation():
#     """Test setting image interpolation mode."""
#     data = np.random.random((10, 15))
#     layer = Image(data)
#     assert layer.interpolation == 'nearest'
#
#     layer = Image(data, interpolation='bicubic')
#     assert layer.interpolation == 'bicubic'
#
#     layer.interpolation = 'bilinear'
#     assert layer.interpolation == 'bilinear'
#
#
# def test_colormaps():
#     """Test setting test_colormaps."""
#     data = np.random.random((10, 15))
#     layer = Image(data)
#     assert layer.colormap[0] == 'gray'
#     assert type(layer.colormap[1]) == Colormap
#
#     layer.colormap = 'magma'
#     assert layer.colormap[0] == 'magma'
#     assert type(layer.colormap[1]) == Colormap
#
#     cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.2, 1.0]])
#     layer.colormap = 'custom', cmap
#     assert layer.colormap[0] == 'custom'
#     assert layer.colormap[1] == cmap
#
#     cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.6, 1.0]])
#     layer.colormap = {'new': cmap}
#     assert layer.colormap[0] == 'new'
#     assert layer.colormap[1] == cmap
#
#     layer = Image(data, colormap='magma')
#     assert layer.colormap[0] == 'magma'
#     assert type(layer.colormap[1]) == Colormap
#
#     cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.2, 1.0]])
#     layer = Image(data, colormap=('custom', cmap))
#     assert layer.colormap[0] == 'custom'
#     assert layer.colormap[1] == cmap
#
#     cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.6, 1.0]])
#     layer = Image(data, colormap={'new': cmap})
#     assert layer.colormap[0] == 'new'
#     assert layer.colormap[1] == cmap
#
#
# def test_clims():
#     """Test setting color limits."""
#     data = np.random.random((10, 15))
#     layer = Image(data)
#     assert layer.clim[0] >= 0
#     assert layer.clim[1] <= 1
#     assert layer.clim[0] < layer.clim[1]
#     assert layer.clim == layer._clim_range
#
#     # Change clim property
#     clim = [0, 2]
#     layer.clim = clim
#     assert layer.clim == clim
#     assert layer._clim_range == clim
#
#     # Set clim as keyword argument
#     layer = Image(data, clim=clim)
#     assert layer.clim == clim
#     assert layer._clim_range == clim
#
#
# def test_clim_range():
#     """Test setting color limits range."""
#     data = np.random.random((10, 15))
#     layer = Image(data)
#     assert layer._clim_range[0] >= 0
#     assert layer._clim_range[1] <= 1
#     assert layer._clim_range[0] < layer._clim_range[1]
#
#     # If all data is the same value the clim_range and clim defaults to [0, 1]
#     data = np.zeros((10, 15))
#     layer = Image(data)
#     assert layer._clim_range == [0, 1]
#     assert layer.clim == [0.0, 1.0]
#
#     # Set clim_range as keyword argument
#     data = np.random.random((10, 15))
#     layer = Image(data, clim_range=[0, 2])
#     assert layer._clim_range == [0, 2]
#
#     # Set clim and clim_range as keyword arguments
#     data = np.random.random((10, 15))
#     layer = Image(data, clim=[0.3, 0.6], clim_range=[0, 2])
#     assert layer.clim == [0.3, 0.6]
#     assert layer._clim_range == [0, 2]
#
#
# def test_metadata():
#     """Test setting image metadata."""
#     data = np.random.random((10, 15))
#     layer = Image(data)
#     assert layer.metadata == {}
#
#     layer = Image(data, metadata={'unit': 'cm'})
#     assert layer.metadata == {'unit': 'cm'}
#
#
# def test_value():
#     """Test getting the value of the data at the current coordinates."""
#     data = np.random.random((10, 15))
#     layer = Image(data)
#     coord, value = layer.get_value()
#     assert np.all(coord == [0, 0])
#     assert value == data[0, 0]
#
#
# def test_message():
#     """Test converting value and coords to message."""
#     data = np.random.random((10, 15))
#     layer = Image(data)
#     coord, value = layer.get_value()
#     msg = layer.get_message(coord, value)
#     assert type(msg) == str
#
#
# def test_thumbnail():
#     """Test the image thumbnail for square data."""
#     data = np.random.random((30, 30))
#     layer = Image(data)
#     layer._update_thumbnail()
#     assert layer.thumbnail.shape == layer._thumbnail_shape
#
#
# def test_xml_list():
#     """Test the xml generation."""
#     data = np.random.random((15, 30))
#     layer = Image(data)
#     xml = layer.to_xml_list()
#     assert type(xml) == list
#     assert len(xml) == 1
#     assert type(xml[0]) == Element
