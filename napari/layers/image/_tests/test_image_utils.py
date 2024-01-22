import time

import dask
import dask.array as da
import numpy as np
import pytest
import skimage
from hypothesis import given
from hypothesis.extra.numpy import array_shapes
from skimage.transform import pyramid_gaussian

from napari.layers.image._image_utils import guess_multiscale, guess_rgb
from napari.layers.image._slice import _ImageSliceRequest
from napari.layers.utils._slice_input import _ThickNDSlice

data_dask = da.random.random(
    size=(100_000, 1000, 1000), chunks=(1, 1000, 1000)
)


def test_guess_rgb():
    shape = (10, 15)
    assert not guess_rgb(shape)

    shape = (10, 15, 6)
    assert not guess_rgb(shape)

    shape = (10, 15, 3)
    assert guess_rgb(shape)

    shape = (10, 15, 4)
    assert guess_rgb(shape)


@given(shape=array_shapes(min_dims=3, min_side=0))
def test_guess_rgb_property(shape):
    assert guess_rgb(shape) == (shape[-1] in (3, 4))


def test_guess_multiscale():
    data = np.random.random((10, 15))
    assert not guess_multiscale(data)[0]

    data = np.random.random((10, 15, 6))
    assert not guess_multiscale(data)[0]

    data = [np.random.random((10, 15, 6))]
    assert not guess_multiscale(data)[0]

    data = [np.random.random((10, 15, 6)), np.random.random((5, 7, 3))]
    assert guess_multiscale(data)[0]

    data = [np.random.random((10, 15, 6)), np.random.random((10, 7, 3))]
    assert guess_multiscale(data)[0]

    data = tuple(data)
    assert guess_multiscale(data)[0]

    if skimage.__version__ > '0.19':
        pyramid_kwargs = {'channel_axis': None}
    else:
        pyramid_kwargs = {'multichannel': False}

    data = tuple(
        pyramid_gaussian(np.random.random((10, 15)), **pyramid_kwargs)
    )
    assert guess_multiscale(data)[0]

    data = np.asarray(
        tuple(pyramid_gaussian(np.random.random((10, 15)), **pyramid_kwargs)),
        dtype=object,
    )
    assert guess_multiscale(data)[0]

    # Check for integer overflow with big data
    s = 8192
    data = [da.ones((s,) * 3), da.ones((s // 2,) * 3), da.ones((s // 4,) * 3)]
    assert guess_multiscale(data)[0]

    # Test for overflow in calculating array sizes
    s = 17179869184
    data = [
        da.from_delayed(
            dask.delayed(lambda: None), shape=(s,) * 2, dtype=np.float64
        ),
        da.from_delayed(
            dask.delayed(lambda: None), shape=(s // 2,) * 2, dtype=np.float64
        ),
    ]
    assert guess_multiscale(data)[0]


def test_guess_multiscale_strip_single_scale():
    data = [np.empty((10, 10))]
    guess, data_out = guess_multiscale(data)
    assert data_out is data[0]
    assert guess is False


def test_guess_multiscale_non_array_list():
    """Check that non-decreasing list input raises ValueError"""
    data = [np.empty((10, 15, 6))] * 2
    with pytest.raises(ValueError):
        _, _ = guess_multiscale(data)


def test_guess_multiscale_incorrect_order():
    data = [np.empty((10, 15)), np.empty((5, 6)), np.empty((20, 15))]
    with pytest.raises(ValueError):
        _, _ = guess_multiscale(data)


def test_timing_multiscale_big():
    now = time.monotonic()
    assert not guess_multiscale(data_dask)[0]
    elapsed = time.monotonic() - now
    assert elapsed < 2, "test was too slow, computation was likely not lazy"


def test_create_data_indexing():
    point = (np.nan, 10.1, 2.6, 4)
    idx = _ImageSliceRequest._point_to_slices(point)
    expected = (slice(None), 10, 3, 4)
    assert idx == expected

    # note that testing entirely out of bounds slices is wrong because these methods
    # assume the bounds check already happened
    data_slice = _ThickNDSlice(
        point=(np.nan, 10.1, 2.6, 4, -1),
        margin_left=(np.nan, 0, 1.6, 0.3, 1),
        margin_right=(np.nan, 0.1, 0.3, 0.5, 0.6),
    )
    idx = _ImageSliceRequest._data_slice_to_slices(
        data_slice, dims_displayed=(0,)
    )
    expected = (
        slice(None),
        slice(10, 11),
        slice(1, 3),
        slice(4, 5),
        slice(0, 1),
    )
    assert idx == expected
