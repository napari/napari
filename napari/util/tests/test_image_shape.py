import pytest
import numpy as np
import dask.array as da
from skimage.transform import pyramid_gaussian
from napari.util.image_shape import (
    guess_rgb,
    get_ndim_and_rgb,
    guess_pyramid,
    should_be_pyramid,
    get_pyramid,
    fast_pyramid,
    trim_pyramid,
    make_pyramid,
)


data_dask = da.random.random(
    size=(100_000, 1000, 1000), chunks=(1, 1000, 1000)
)


def test_guess_rgb():
    # Test 2D image
    shape = (10, 15)
    assert not guess_rgb(shape)

    # Test 3D image that cannot be rgb
    shape = (10, 15, 6)
    assert not guess_rgb(shape)

    # Test image that could be rgb
    shape = (10, 15, 3)
    assert guess_rgb(shape)

    # Test image that could be rgba
    shape = (10, 15, 4)
    assert guess_rgb(shape)


def test_get_ndim_and_rgb():
    # Test 2D image
    shape = (10, 15)
    ndim, rgb = get_ndim_and_rgb(shape, False)
    assert not rgb
    assert ndim == 2

    ndim, rgb = get_ndim_and_rgb(shape, None)
    assert not rgb
    assert ndim == 2

    with pytest.raises(ValueError):
        ndim, rgb = get_ndim_and_rgb(shape, True)

    # Test 3D image that cannot be rgb
    shape = (10, 15, 6)
    ndim, rgb = get_ndim_and_rgb(shape, False)
    assert not rgb
    assert ndim == 3

    ndim, rgb = get_ndim_and_rgb(shape, None)
    assert not rgb
    assert ndim == 3

    with pytest.raises(ValueError):
        ndim, rgb = get_ndim_and_rgb(shape, True)

    # Test image that could be rgb
    shape = (10, 15, 3)
    ndim, rgb = get_ndim_and_rgb(shape, False)
    assert not rgb
    assert ndim == 3

    ndim, rgb = get_ndim_and_rgb(shape, None)
    assert rgb
    assert ndim == 2

    ndim, rgb = get_ndim_and_rgb(shape, True)
    assert rgb
    assert ndim == 2

    # Test image that could be rgba
    shape = (10, 15, 4)
    ndim, rgb = get_ndim_and_rgb(shape, False)
    assert not rgb
    assert ndim == 3

    ndim, rgb = get_ndim_and_rgb(shape, None)
    assert rgb
    assert ndim == 2

    ndim, rgb = get_ndim_and_rgb(shape, True)


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
def test_timing_guess_pyramid_big():
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
    assert np.all(should_be_pyramid(shape) == [False] * 2)

    shape = (10, 15, 6)
    assert np.all(should_be_pyramid(shape) == [False] * 3)

    shape = (16_0000, 15, 3)
    assert np.all(should_be_pyramid(shape) == [True] + [False] * 2)

    shape = (2 ** 13 + 1, 100, 3)
    assert np.all(should_be_pyramid(shape) == [True] + [False] * 2)

    shape = (2 ** 13, 100, 4)
    assert np.all(should_be_pyramid(shape) == [False] * 3)


def test_fast_pyramid():
    shape = (64, 64)
    data = np.random.random(shape)
    pyramid = fast_pyramid(data)
    assert np.all(pyramid[0] == data)
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
    assert np.all(pyramid[0] == data)
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
    assert np.all(pyramid[0] == data)
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
    assert np.all(pyramid[0] == data)
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
    assert np.all(pyramid[0] == data)
    assert np.all(
        [
            p.shape == (shape[0] // 2 ** i, shape[1] // 2 ** i, 3)
            for i, p in enumerate(pyramid)
        ]
    )
    assert len(pyramid) == 7


def test_make_pyramid():
    shape = (64, 64)
    data = np.random.random(shape)
    pyramid = make_pyramid(data, [True, True])
    assert len(pyramid) > 0
    assert np.all(pyramid[0] == data)
    assert np.all(
        [
            p.shape == (shape[0] // 2 ** i, shape[1] // 2 ** i)
            for i, p in enumerate(pyramid)
        ]
    )

    shape = (64, 64, 3)
    data = np.random.random(shape)
    pyramid = make_pyramid(data, [True, True, False])
    assert len(pyramid) > 0
    assert np.all(pyramid[0] == data)
    assert np.all(
        [
            p.shape == (shape[0] // 2 ** i, shape[1] // 2 ** i, 3)
            for i, p in enumerate(pyramid)
        ]
    )

    shape = (64, 64, 3)
    data = np.random.random(shape)
    pyramid = make_pyramid(data, [True, False, False])
    assert len(pyramid) > 0
    assert np.all(pyramid[0] == data)
    assert np.all(
        [
            p.shape == (shape[0] // 2 ** i, shape[1], 3)
            for i, p in enumerate(pyramid)
        ]
    )


def test_get_pyramid():
    data = np.random.random((10, 15))
    data_pyramid = get_pyramid(data)
    assert data_pyramid is None

    data_pyramid = get_pyramid(data, is_pyramid=False)
    assert data_pyramid is None

    data = [np.random.random((10, 15, 6)), np.random.random((5, 7, 3))]
    data_pyramid = get_pyramid(data)
    assert np.all([np.all(dp == d) for dp, d in zip(data_pyramid, data)])

    data_pyramid = get_pyramid(data, is_pyramid=True)
    assert np.all([np.all(dp == d) for dp, d in zip(data_pyramid, data)])

    shape = (20_000, 15)
    data = np.random.random(shape)
    data_pyramid = get_pyramid(data, is_pyramid=False)
    assert data_pyramid is None

    data_pyramid = get_pyramid(data, force_pyramid=True)
    assert len(data_pyramid) > 0
    assert np.all(data_pyramid[0] == data)
    assert np.all(
        [
            p.shape == (shape[0] // 2 ** i, shape[1])
            for i, p in enumerate(data_pyramid)
        ]
    )

    data_pyramid = get_pyramid(data, force_pyramid=[True, False])
    assert len(data_pyramid) > 0
    assert np.all(data_pyramid[0] == data)
    assert np.all(
        [
            p.shape == (shape[0] // 2 ** i, shape[1])
            for i, p in enumerate(data_pyramid)
        ]
    )

    with pytest.raises(ValueError):
        data_pyramid = get_pyramid(data, is_pyramid=False, force_pyramid=True)
