import dask.array as da
from napari.layers import Image
import pytest


data = da.random.random(size=(100_000, 1000, 1000), chunks=(1, 1000, 1000))


@pytest.mark.timeout(2)
def test_fast_big_dask_all_specified():
    layer = Image(data, is_pyramid=False, contrast_limits=[0, 1])
    assert layer.data.shape == data.shape


@pytest.mark.timeout(2)
def test_fast_big_dask_is_pyramid_specified():
    layer = Image(data, is_pyramid=False)
    assert layer.data.shape == data.shape


@pytest.mark.skip(reason="currently fails because of `guess_pyramid`")
@pytest.mark.timeout(2)
def test_fast_big_dask_contrast_limits_specified():
    layer = Image(data, contrast_limits=[0, 1])
    assert layer.data.shape == data.shape


@pytest.mark.skip(reason="currently fails because of `guess_pyramid`")
@pytest.mark.timeout(2)
def test_fast_big_dask_nothing_specified():
    layer = Image(data)
    assert layer.data.shape == data.shape
