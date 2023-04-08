import numpy as np
import pytest

from napari._tests.utils import layer_test_data
from napari.layers import Image, Labels


@pytest.mark.parametrize(
    'image_shape, dims_displayed, expected',
    [
        ((10, 20, 30), (0, 1, 2), [[0, 10], [0, 20], [0, 30]]),
        ((10, 20, 30), (0, 2, 1), [[0, 10], [0, 30], [0, 20]]),
        ((10, 20, 30), (2, 1, 0), [[0, 30], [0, 20], [0, 10]]),
    ],
)
def test_layer_bounding_box_order(image_shape, dims_displayed, expected):
    layer = Image(data=np.random.random(image_shape))
    #
    assert np.allclose(
        layer._display_bounding_box(dims_displayed=dims_displayed), expected
    )


@pytest.mark.parametrize('Layer, data, ndim', layer_test_data)
def test_update_scale_updates_layer_extent_cache(Layer, data, ndim):
    np.random.seed(0)
    layer = Layer(data)
    # Check layer has been correctly created
    assert layer.ndim == ndim
    np.testing.assert_almost_equal(layer.extent.step, (1,) * layer.ndim)

    # Check layer extent change when scale changes
    old_extent = layer.extent
    layer.scale = (2,) * layer.ndim
    new_extent = layer.extent
    assert old_extent is not layer.extent
    assert new_extent is layer.extent
    np.testing.assert_almost_equal(layer.extent.step, (2,) * layer.ndim)


@pytest.mark.parametrize('Layer, data, ndim', layer_test_data)
def test_update_data_updates_layer_extent_cache(Layer, data, ndim):
    np.random.seed(0)
    layer = Layer(data)
    # Check layer has been correctly created
    assert layer.ndim == ndim

    # Check layer extent change when data changes
    old_extent = layer.extent
    try:
        layer.data = data + 1
    except TypeError:
        return
    new_extent = layer.extent
    assert old_extent is not layer.extent
    assert new_extent is layer.extent


def test_contrast_limits_must_be_increasing():
    np.random.seed(0)
    Image(np.random.rand(8, 8), contrast_limits=[0, 1])
    with pytest.raises(ValueError):
        Image(np.random.rand(8, 8), contrast_limits=[1, 1])
    with pytest.raises(ValueError):
        Image(np.random.rand(8, 8), contrast_limits=[1, 0])


def _check_subpixel_values(layer, val_dict):
    ndisplay = layer._slice_input.ndisplay
    for center, expected_value in val_dict.items():
        # ensure all positions within the pixel extent report the same value
        # note: values are checked in data coordinates in this function
        for offset_0 in [-0.4999, 0, 0.4999]:
            for offset_1 in [-0.4999, 0, 0.4999]:
                position = [center[0] + offset_0, center[1] + offset_1]
                view_direction = None
                dims_displayed = None
                if ndisplay == 3:
                    position = [0, *position]
                    if isinstance(layer, Labels):
                        # Labels implements _get_value_3d, Image does not
                        view_direction = np.asarray([1.0, 0, 0])
                        dims_displayed = [0, 1, 2]

                val = layer.get_value(
                    position=position,
                    view_direction=view_direction,
                    dims_displayed=dims_displayed,
                    world=False,
                )
                assert val == expected_value


@pytest.mark.parametrize('ImageClass', [Image, Labels])
@pytest.mark.parametrize('ndim', [2, 3])
def test_get_value_at_subpixel_offsets(ImageClass, ndim):
    """check value at various shifts within a pixel/voxel's extent"""
    if ndim == 3:
        data = np.arange(1, 9).reshape(2, 2, 2)
    elif ndim == 2:
        data = np.arange(1, 5).reshape(2, 2)

    # test using non-uniform scale per-axis
    layer = ImageClass(data, scale=(0.5, 1, 2)[:ndim])
    layer._slice_dims([0] * ndim, ndisplay=ndim)

    # dictionary of expected values at each voxel center coordinate
    val_dict = {
        (0, 0): data[(0,) * (ndim - 2) + (0, 0)],
        (0, 1): data[(0,) * (ndim - 2) + (0, 1)],
        (1, 0): data[(0,) * (ndim - 2) + (1, 0)],
        (1, 1): data[(0,) * (ndim - 2) + (1, 1)],
    }
    _check_subpixel_values(layer, val_dict)


@pytest.mark.parametrize('ImageClass', [Image, Labels])
def test_get_value_3d_view_of_2d_image(ImageClass):
    """check value at various shifts within a pixel/voxel's extent"""
    data = np.arange(1, 5).reshape(2, 2)

    ndisplay = 3
    # test using non-uniform scale per-axis
    layer = ImageClass(data, scale=(0.5, 1))
    layer._slice_dims([0] * ndisplay, ndisplay=ndisplay)

    # dictionary of expected values at each voxel center coordinate
    val_dict = {
        (0, 0): data[(0, 0)],
        (0, 1): data[(0, 1)],
        (1, 0): data[(1, 0)],
        (1, 1): data[(1, 1)],
    }
    _check_subpixel_values(layer, val_dict)


def test_zero_scale_layer():
    with pytest.raises(ValueError, match='scale values of 0'):
        Image(np.zeros((64, 64)), scale=(0, 1))
