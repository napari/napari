import dask.array as da
import numpy as np
import pytest
from skimage.transform import pyramid_gaussian
from napari.layers.image.image_utils import (
    fast_pyramid,
    get_pyramid_and_rgb,
    guess_pyramid,
    guess_rgb,
    should_be_pyramid,
    trim_pyramid,
)


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


def test_guess_pyramid():
    data = np.random.random((10, 15))
    assert not guess_pyramid(data)

    data = np.random.random((10, 15, 6))
    assert not guess_pyramid(data)

    data = [np.random.random((10, 15, 6))]
    assert not guess_pyramid(data)

    data = [np.random.random((10, 15, 6)), np.random.random((10, 15, 6))]
    assert not guess_pyramid(data)

    data = [np.random.random((10, 15, 6)), np.random.random((5, 7, 3))]
    assert guess_pyramid(data)

    data = [np.random.random((10, 15, 6)), np.random.random((10, 7, 3))]
    assert guess_pyramid(data)

    data = tuple(data)
    assert guess_pyramid(data)

    data = tuple(
        pyramid_gaussian(np.random.random((10, 15)), multichannel=False)
    )
    assert guess_pyramid(data)

    data = np.asarray(
        tuple(pyramid_gaussian(np.random.random((10, 15)), multichannel=False))
    )
    assert guess_pyramid(data)

    # Check for integer overflow with big data
    s = 8192
    data = [da.ones((s,) * 3), da.ones((s // 2,) * 3), da.ones((s // 4,) * 3)]
    assert guess_pyramid(data)


@pytest.mark.timeout(2)
def test_timing_is_pyramid_big():
    assert not guess_pyramid(data_dask)


def test_trim_pyramid():

    data = [np.random.random((20, 30)), np.random.random((10, 15))]
    trimmed = trim_pyramid(data)
    assert np.all([np.all(t == d) for t, d in zip(data, trimmed)])

    data = [
        np.random.random((40, 60)),
        np.random.random((20, 30)),
        np.random.random((10, 15)),
    ]
    trimmed = trim_pyramid(data)
    assert np.all([np.all(t == d) for t, d in zip(data[:2], trimmed)])

    data = [
        np.random.random((400, 10)),
        np.random.random((200, 10)),
        np.random.random((100, 10)),
        np.random.random((50, 10)),
    ]
    trimmed = trim_pyramid(data)
    assert np.all([np.all(t == d) for t, d in zip(data[:3], trimmed)])


def test_should_be_pyramid():
    shape = (10, 15)
    assert not np.any(should_be_pyramid(shape))

    shape = (10, 15, 6)
    assert not np.any(should_be_pyramid(shape))

    shape = (16_0000, 15, 3)
    assert np.any(should_be_pyramid(shape))

    shape = (2 ** 13, 100, 3)
    assert np.any(should_be_pyramid(shape))

    shape = (2 ** 13 - 1, 100, 4)
    assert not np.any(should_be_pyramid(shape))


def test_get_pyramid_and_rgb():
    data = np.random.random((10, 15))
    ndim, rgb, pyramid, data_pyramid = get_pyramid_and_rgb(data)
    assert not pyramid
    assert data_pyramid is None
    assert not rgb
    assert ndim == 2

    data = np.random.random((80, 40))
    ndim, rgb, pyramid, data_pyramid = get_pyramid_and_rgb(data, pyramid=True)
    assert pyramid
    assert data_pyramid[0].shape == (80, 40)
    assert not rgb
    assert ndim == 2

    data = [np.random.random((10, 15, 6)), np.random.random((5, 7, 3))]
    ndim, rgb, pyramid, data_pyramid = get_pyramid_and_rgb(data)
    assert pyramid
    assert np.all([np.all(dp == d) for dp, d in zip(data_pyramid, data)])
    assert not rgb
    assert ndim == 3

    shape = (20_000, 20)
    data = np.random.random(shape)
    ndim, rgb, pyramid, data_pyramid = get_pyramid_and_rgb(data)
    assert pyramid
    assert data_pyramid[0].shape == shape
    assert data_pyramid[1].shape == (shape[0] / 2, shape[1])
    assert not rgb
    assert ndim == 2

    ndim, rgb, pyramid, data_pyramid = get_pyramid_and_rgb(data, pyramid=False)
    assert not pyramid
    assert data_pyramid is None
    assert not rgb
    assert ndim == 2


def test_fast_pyramid():
    shape = (64, 64)
    data = np.random.random(shape)
    pyramid = fast_pyramid(data)
    assert np.all(
        [
            p.shape == (shape[0] // 2 ** i, shape[1] // 2 ** i)
            for i, p in enumerate(pyramid)
        ]
    )
    assert len(pyramid) == 7

    shape = (64, 64)
    data = np.random.random(shape)
    pyramid = fast_pyramid(data, max_layer=3)
    assert np.all(
        [
            p.shape == (shape[0] // 2 ** i, shape[1] // 2 ** i)
            for i, p in enumerate(pyramid)
        ]
    )
    assert len(pyramid) == 3

    shape = (64, 16)
    data = np.random.random(shape)
    pyramid = fast_pyramid(data, downscale=(2, 1))
    assert np.all(
        [
            p.shape == (shape[0] // 2 ** i, shape[1])
            for i, p in enumerate(pyramid)
        ]
    )
    assert len(pyramid) == 7

    shape = (64, 64, 3)
    data = np.random.random(shape)
    pyramid = fast_pyramid(data, downscale=(2, 2, 1))
    assert np.all(
        [
            p.shape == (shape[0] // 2 ** i, shape[1] // 2 ** i, 3)
            for i, p in enumerate(pyramid)
        ]
    )
    assert len(pyramid) == 7

    shape = (64, 32, 3)
    data = np.random.random(shape)
    pyramid = fast_pyramid(data, downscale=(2, 2, 1))
    assert np.all(
        [
            p.shape == (shape[0] // 2 ** i, shape[1] // 2 ** i, 3)
            for i, p in enumerate(pyramid)
        ]
    )
    assert len(pyramid) == 7
