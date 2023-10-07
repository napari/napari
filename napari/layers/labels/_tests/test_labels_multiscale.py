import numpy as np

from napari.layers import Labels

from napari.components.dims import Dims


def test_random_multiscale():
    """Test instantiating Labels layer with random 2D multiscale data."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.randint(20, size=s) for s in shapes]
    layer = Labels(data, multiscale=True)
    assert layer.data == data
    assert layer.multiscale is True
    assert layer.editable is False
    assert layer.ndim == len(shapes[0])
    np.testing.assert_array_equal(
        layer.extent.data[1], [s - 1 for s in shapes[0]]
    )
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_infer_multiscale():
    """Test instantiating Labels layer with random 2D multiscale data."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.randint(20, size=s) for s in shapes]
    layer = Labels(data)
    assert layer.data == data
    assert layer.multiscale is True
    assert layer.editable is False
    assert layer.ndim == len(shapes[0])
    np.testing.assert_array_equal(
        layer.extent.data[1], [s - 1 for s in shapes[0]]
    )
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_3D_multiscale_labels_in_2D():
    """Test instantiating Labels layer with 3D data, 2D dims."""
    data_multiscale, layer = instantiate_3D_multiscale_labels()

    assert layer.data == data_multiscale
    assert layer.multiscale is True
    assert layer.editable is False
    assert layer.ndim == len(data_multiscale[0].shape)
    np.testing.assert_array_equal(
        layer.extent.data[1], np.array(data_multiscale[0].shape)-1
    )
    assert layer.rgb is False
    assert layer._data_view.ndim == 2

    # check corner pixels, should be tuple of highest resolution level
    assert layer.get_value([0,0,0]) == (layer.data_level, data_multiscale[0][0,0,0])

def test_3D_multiscale_labels_in_3D():
    """Test instantiating Labels layer with 3D data, 3D dims."""
    data_multiscale, layer = instantiate_3D_multiscale_labels()

    # use 3D dims
    layer._slice_dims(Dims(ndim=3, ndisplay=3))
    assert layer._data_view.ndim == 3

    # check corner pixels, should be value of lowest resolution level
    assert layer.get_value([0, 0, 0], view_direction=[1, 0, 0], dims_displayed=[0, 1, 2]) == 1
    # assert layer.get_value([0, 9, 9], view_direction=[1, 0, 0], dims_displayed=[0, 1, 2]) == 2
    # assert layer.get_value([9, 9, 9], view_direction=[1, 0, 0], dims_displayed=[0, 1, 2]) == 2


def instantiate_3D_multiscale_labels():
    data = np.arange(1000).reshape((10, 10, 10))
    data_multiscale = [data, data[::2, ::2, ::2]]

    return data_multiscale, Labels(data_multiscale, multiscale=True)