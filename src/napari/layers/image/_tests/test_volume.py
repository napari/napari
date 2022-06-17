import numpy as np

from napari.layers import Image
from napari.layers.image._image_mouse_bindings import move_plane_along_normal


def test_random_volume():
    """Test instantiating Image layer with random 3D data."""
    shape = (10, 15, 20)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    layer._slice_dims(ndisplay=3)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], shape)
    assert layer._data_view.shape == shape[-3:]


def test_switching_displayed_dimensions():
    """Test instantiating data then switching to displayed."""
    shape = (10, 15, 20)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], shape)

    # check displayed data is initially 2D
    assert layer._data_view.shape == shape[-2:]

    layer._slice_dims(ndisplay=3)
    # check displayed data is now 3D
    assert layer._data_view.shape == shape[-3:]

    layer._slice_dims(ndisplay=2)
    # check displayed data is now 2D
    assert layer._data_view.shape == shape[-2:]

    layer = Image(data)
    layer._slice_dims(ndisplay=3)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], shape)

    # check displayed data is initially 3D
    assert layer._data_view.shape == shape[-3:]

    layer._slice_dims(ndisplay=2)
    # check displayed data is now 2D
    assert layer._data_view.shape == shape[-2:]

    layer._slice_dims(ndisplay=3)
    # check displayed data is now 3D
    assert layer._data_view.shape == shape[-3:]


def test_all_zeros_volume():
    """Test instantiating Image layer with all zeros data."""
    shape = (10, 15, 20)
    data = np.zeros(shape, dtype=float)
    layer = Image(data)
    layer._slice_dims(ndisplay=3)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], shape)
    assert layer._data_view.shape == shape[-3:]


def test_integer_volume():
    """Test instantiating Image layer with integer data."""
    shape = (10, 15, 20)
    np.random.seed(0)
    data = np.round(10 * np.random.random(shape)).astype(int)
    layer = Image(data)
    layer._slice_dims(ndisplay=3)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], shape)
    assert layer._data_view.shape == shape[-3:]


def test_3D_volume():
    """Test instantiating Image layer with random 3D data."""
    shape = (10, 15, 6)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    layer._slice_dims(ndisplay=3)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], shape)
    assert layer._data_view.shape == shape[-3:]


def test_4D_volume():
    """Test instantiating multiple Image layers with random 4D data."""
    shape = (10, 15, 6, 8)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    layer._slice_dims(ndisplay=3)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], shape)
    assert layer._data_view.shape == shape[-3:]


def test_changing_volume():
    """Test changing Image data."""
    shape_a = (10, 15, 30)
    shape_b = (20, 12, 6)
    np.random.seed(0)
    data_a = np.random.random(shape_a)
    data_b = np.random.random(shape_b)
    layer = Image(data_a)
    layer._slice_dims(ndisplay=3)
    layer.data = data_b
    assert np.all(layer.data == data_b)
    assert layer.ndim == len(shape_b)
    np.testing.assert_array_equal(layer.extent.data[1], shape_b)
    assert layer._data_view.shape == shape_b[-3:]


def test_scale():
    """Test instantiating anisotropic 3D volume."""
    shape = (10, 15, 20)
    scale = [3, 1, 1]
    full_shape = tuple(np.multiply(shape, scale))
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data, scale=scale)
    layer._slice_dims(ndisplay=3)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(
        layer.extent.world[1] - layer.extent.world[0], full_shape
    )
    pixel_extent_end = np.asarray(full_shape) - 0.5 * np.asarray(scale)
    np.testing.assert_array_equal(layer.extent.world[1], pixel_extent_end)
    # Note that the scale appears as the step size in the range
    assert layer._data_view.shape == shape[-3:]


def test_value():
    """Test getting the value of the data at the current coordinates."""
    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    layer = Image(data)
    layer._slice_dims(ndisplay=3)
    value = layer.get_value((0,) * 3)
    assert value == data[0, 0, 0]


def test_message():
    """Test converting value and coords to message."""
    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    layer = Image(data)
    layer._slice_dims(ndisplay=3)
    msg = layer.get_status((0,) * 3)
    assert type(msg) == str


def test_plane_drag_callback():
    """Plane drag callback should only be active when depicting as plane."""
    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    layer = Image(data, depiction='volume')

    assert move_plane_along_normal not in layer.mouse_drag_callbacks
    layer.depiction = 'plane'
    assert move_plane_along_normal in layer.mouse_drag_callbacks
    layer.depiction = 'volume'
    assert move_plane_along_normal not in layer.mouse_drag_callbacks
