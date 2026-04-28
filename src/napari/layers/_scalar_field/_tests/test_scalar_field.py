import numpy as np
import pytest

from napari.components import Dims
from napari.layers import Image, Labels
from napari.layers._scalar_field.scalar_field import ScalarFieldBase
from napari.utils._test_utils import (
    validate_all_params_in_docstring,
    validate_docstring_parent_class_consistency,
    validate_kwargs_sorted,
)


def test_docstring():
    validate_all_params_in_docstring(ScalarFieldBase)
    validate_kwargs_sorted(ScalarFieldBase)
    validate_docstring_parent_class_consistency(
        ScalarFieldBase, skip=('data', 'ndim', 'multiscale')
    )


def test_multiscale_thumbnail_level_prematerialized():
    """Thumbnail level materializes as numpy via the level materializer."""
    rng = np.random.default_rng(0)
    data = [rng.random((128, 128)), rng.random((64, 64))]
    layer = Image(data, multiscale=True)
    result = layer._level_materializer(layer._thumbnail_level)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, data[layer._thumbnail_level])


def test_3d_multiscale_thumbnail_not_prematerialized():
    """3D multiscale images should not create a level materializer: volumes can be too large."""
    data = [np.random.random((16, 128, 128)), np.random.random((8, 64, 64))]
    layer = Image(data, multiscale=True)
    assert layer._level_materializer is None


def test_single_scale_no_prematerialization():
    """Single-scale images should not materialize and cache."""
    layer = Image(np.random.random((32, 32)))
    assert layer._level_materializer is None


def test_multiscale_slice_request_receives_preselected_sources():
    """Slice requests should receive the preselected sources."""
    data = [np.random.random((64, 64)), np.random.random((32, 32))]
    layer = Image(data, multiscale=True)
    dims = Dims(
        ndim=2,
        ndisplay=2,
        range=tuple((0, s - 1, 1) for s in data[0].shape),
    )
    request = layer._slicing_state._make_slice_request(dims)
    assert layer.data_level == layer._thumbnail_level
    # slice request should re-use thumbnail source when possible
    assert request.data_at_data_level is request.data_at_thumbnail_level
    np.testing.assert_array_equal(
        request.data_at_thumbnail_level, data[layer._thumbnail_level]
    )


def test_multiscale_slice_request_keeps_non_thumbnail_image_source():
    """Non-thumbnail views should keep their own level."""
    data = [np.random.random((64, 64)), np.random.random((32, 32))]
    layer = Image(data, multiscale=True)
    layer.data_level = 0
    dims = Dims(
        ndim=2,
        ndisplay=2,
        range=tuple((0, s - 1, 1) for s in data[0].shape),
    )
    request = layer._slicing_state._make_slice_request(dims)
    assert request.data_at_data_level is data[0]
    np.testing.assert_array_equal(
        request.data_at_thumbnail_level, data[layer._thumbnail_level]
    )


def test_thumbnail_level_data_refreshed_on_data_replacement():
    """Replacing layer.data must bind a fresh materializer for the new data
    so slices never read from the old dataset."""
    data1 = [np.zeros((64, 64)), np.zeros((32, 32))]
    data2 = [np.ones((90, 90)), np.ones((45, 45)), np.ones((22, 22))]

    layer = Image(data1, multiscale=True)
    assert layer._thumbnail_level == 1
    np.testing.assert_array_equal(
        layer._level_materializer(layer._thumbnail_level), data1[1]
    )

    layer.data = data2
    assert layer._thumbnail_level == 2  # new last level
    np.testing.assert_array_equal(
        layer._level_materializer(layer._thumbnail_level), data2[2]
    )


def test_thumbnail_level_data_uses_new_data_ndim_2d_to_3d():
    """Replacing 2D data with 3D should set materializer to None — 3D volumes
    are left as lazy data-array references."""
    from napari.layers._multiscale_data import MultiScaleData

    data2d = [np.zeros((64, 64)), np.zeros((32, 32))]
    data3d = [np.ones((16, 64, 64)), np.ones((8, 32, 32))]

    layer = Image(data2d, multiscale=True)
    assert layer._level_materializer is not None  # 2D: materializer present

    layer._data = MultiScaleData(data3d)
    layer._reset_thumbnail_level_data()

    assert layer._level_materializer is None  # 3D: no materializer


def test_thumbnail_level_data_uses_new_data_ndim_3d_to_2d():
    """Replacing 3D data with 2D should produce a fresh materializer for
    the new 2D data."""
    from napari.layers._multiscale_data import MultiScaleData

    data3d = [np.zeros((16, 64, 64)), np.zeros((8, 32, 32))]
    data2d = [np.ones((64, 64)), np.ones((32, 32))]

    layer = Image(data3d, multiscale=True)
    old_materializer = layer._level_materializer

    layer._data = MultiScaleData(data2d)
    layer._reset_thumbnail_level_data()
    assert layer._level_materializer is not old_materializer
    np.testing.assert_array_equal(
        layer._level_materializer(layer._thumbnail_level),
        data2d[layer._thumbnail_level],
    )


def test_rgb_2d_multiscale_prematerialized():
    """RGB (H, W, 3) multiscale images materialise correctly via the closure."""
    data = [np.ones((128, 128, 3)), np.ones((64, 64, 3))]
    layer = Image(data, multiscale=True, rgb=True)

    assert layer.ndim == 2  # spatial dims only
    tld = layer._level_materializer(layer._thumbnail_level)
    assert isinstance(tld, np.ndarray)
    assert tld.shape == (64, 64, 3)
    np.testing.assert_array_equal(tld, data[1])


@pytest.mark.parametrize('Layer', [Image, Labels])
def test_data_setter_updates_transforms(Layer):
    """Replacing data with different ndim should expand transforms."""
    data_2d = np.zeros((10, 10), dtype=np.uint8)
    layer = Layer(data_2d, scale=(2, 3))
    assert layer.ndim == 2
    assert len(layer.scale) == 2

    data_3d = np.zeros((5, 10, 10), dtype=np.uint8)
    layer.data = data_3d
    assert layer.ndim == 3
    assert len(layer.scale) == 3
