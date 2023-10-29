import numpy as np

from napari.components.dims import Dims
from napari.layers import Labels


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
        layer.extent.data[1], np.array(data_multiscale[0].shape) - 1
    )
    assert layer.rgb is False
    assert layer._data_view.ndim == 2

    # check corner pixels, should be tuple of highest resolution level
    assert layer.get_value([0, 0, 0]) == (
        layer.data_level,
        data_multiscale[0][0, 0, 0],
    )


def test_3D_multiscale_labels_in_3D():
    """Test instantiating Labels layer with 3D data, 3D dims."""
    data_multiscale, layer = instantiate_3D_multiscale_labels()

    # use 3D dims
    layer._slice_dims(Dims(ndim=3, ndisplay=3))
    assert layer._data_view.ndim == 3

    # check corner pixels, should be value of lowest resolution level
    # [0,0,0] has value 0, which is transparent, so the ray will hit the next point
    # which is [1, 0, 0] and has value 4
    # the position array is in original data coords (no downsampling)
    assert (
        layer.get_value(
            [0, 0, 0], view_direction=[1, 0, 0], dims_displayed=[0, 1, 2]
        )
        == 4
    )
    assert (
        layer.get_value(
            [0, 0, 0], view_direction=[-1, 0, 0], dims_displayed=[0, 1, 2]
        )
        == 4
    )
    assert (
        layer.get_value(
            [0, 1, 1], view_direction=[1, 0, 0], dims_displayed=[0, 1, 2]
        )
        == 4
    )
    assert (
        layer.get_value(
            [0, 5, 5], view_direction=[1, 0, 0], dims_displayed=[0, 1, 2]
        )
        == 3
    )
    assert (
        layer.get_value(
            [5, 0, 5], view_direction=[0, 0, -1], dims_displayed=[0, 1, 2]
        )
        == 5
    )


def instantiate_3D_multiscale_labels():
    lowest_res_scale = np.arange(8).reshape(2, 2, 2)
    middle_res_scale = (
        lowest_res_scale.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
    )
    highest_res_scale = (
        middle_res_scale.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
    )

    data_multiscale = [highest_res_scale, middle_res_scale, lowest_res_scale]

    return data_multiscale, Labels(data_multiscale, multiscale=True)
