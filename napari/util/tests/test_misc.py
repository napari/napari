import numpy as np
from napari.util.misc import (
    is_rgb,
    callsignature,
    is_pyramid,
    should_be_pyramid,
    get_pyramid_and_rgb,
    fast_pyramid,
)


def test_is_rgb():
    shape = (10, 15)
    assert not is_rgb(shape)

    shape = (10, 15, 6)
    assert not is_rgb(shape)

    shape = (10, 15, 3)
    assert is_rgb(shape)

    shape = (10, 15, 4)
    assert is_rgb(shape)


def test_is_pyramid():
    data = np.random.random((10, 15))
    assert not is_pyramid(data)

    data = np.random.random((10, 15, 6))
    assert not is_pyramid(data)

    data = [np.random.random((10, 15, 6))]
    assert not is_pyramid(data)

    data = [np.random.random((10, 15, 6)), np.random.random((10, 15, 6))]
    assert not is_pyramid(data)

    data = [np.random.random((10, 15, 6)), np.random.random((5, 7, 3))]
    assert is_pyramid(data)

    data = [np.random.random((10, 15, 6)), np.random.random((10, 7, 3))]
    assert is_pyramid(data)


def test_should_be_pyramid():
    shape = (10, 15)
    assert not np.any(should_be_pyramid(shape))

    shape = (10, 15, 6)
    assert not np.any(should_be_pyramid(shape))

    shape = (16_0000, 15, 3)
    assert np.any(should_be_pyramid(shape))

    shape = (2 ** 13 + 1, 100, 3)
    assert np.any(should_be_pyramid(shape))

    shape = (2 ** 13 - 1, 100, 4)
    assert not np.any(should_be_pyramid(shape))


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


def test_get_pyramid_and_rgb():
    data = np.random.random((10, 15))
    ndim, rgb, pyramid, data_pyramid = get_pyramid_and_rgb(data)
    assert not pyramid
    assert data_pyramid is None
    assert not rgb
    assert ndim == 2

    data = [np.random.random((10, 15, 6)), np.random.random((5, 7, 3))]
    ndim, rgb, pyramid, data_pyramid = get_pyramid_and_rgb(data)
    assert pyramid
    assert np.all([np.all(dp == d) for dp, d in zip(data_pyramid, data)])
    assert not rgb
    assert ndim == 3

    shape = (20_000, 20_000)
    data = np.random.random(shape)
    ndim, rgb, pyramid, data_pyramid = get_pyramid_and_rgb(data)
    assert pyramid
    assert data_pyramid[0].shape == shape
    assert data_pyramid[1].shape == (shape[0] / 2, shape[1] / 2)
    assert not rgb
    assert ndim == 2


def test_callsignature():
    # no arguments
    assert str(callsignature(lambda: None)) == '()'

    # one arg
    assert str(callsignature(lambda a: None)) == '(a)'

    # multiple args
    assert str(callsignature(lambda a, b: None)) == '(a, b)'

    # arbitrary args
    assert str(callsignature(lambda *args: None)) == '(*args)'

    # arg + arbitrary args
    assert str(callsignature(lambda a, *az: None)) == '(a, *az)'

    # default arg
    assert str(callsignature(lambda a=42: None)) == '(a=a)'

    # multiple default args
    assert str(callsignature(lambda a=0, b=1: None)) == '(a=a, b=b)'

    # arg + default arg
    assert str(callsignature(lambda a, b=42: None)) == '(a, b=b)'

    # arbitrary kwargs
    assert str(callsignature(lambda **kwargs: None)) == '(**kwargs)'

    # default arg + arbitrary kwargs
    assert str(callsignature(lambda a=42, **kwargs: None)) == '(a=a, **kwargs)'

    # arg + default arg + arbitrary kwargs
    assert str(callsignature(lambda a, b=42, **kw: None)) == '(a, b=b, **kw)'

    # arbitary args + arbitrary kwargs
    assert str(callsignature(lambda *args, **kw: None)) == '(*args, **kw)'

    # arg + default arg + arbitrary kwargs
    assert (
        str(callsignature(lambda a, b=42, *args, **kwargs: None))
        == '(a, b=b, *args, **kwargs)'
    )

    # kwonly arg
    assert str(callsignature(lambda *, a: None)) == '(a=a)'

    # arg + kwonly arg
    assert str(callsignature(lambda a, *, b: None)) == '(a, b=b)'

    # default arg + kwonly arg
    assert str(callsignature(lambda a=42, *, b: None)) == '(a=a, b=b)'

    # kwonly args + everything
    assert (
        str(callsignature(lambda a, b=42, *, c, d=5, **kwargs: None))
        == '(a, b=b, c=c, d=d, **kwargs)'
    )
