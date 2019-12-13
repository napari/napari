import dask.array as da
import zarr
from napari.layers import Image
import pytest


data_dask = da.random.random(
    size=(100_000, 1000, 1000), chunks=(1, 1000, 1000)
)
data_zarr = zarr.zeros((100_000, 1000, 1000))
data_dask_2D = da.random.random((100_000, 100_000))


@pytest.mark.timeout(2)
@pytest.mark.parametrize('data', [data_dask, data_zarr])
def test_timing_fast_big_dask_all_specified_(data):
    layer = Image(data, is_pyramid=False, contrast_limits=[0, 1])
    assert layer.data.shape == data.shape


@pytest.mark.timeout(2)
@pytest.mark.parametrize('data', [data_dask, data_zarr])
def test_timing_fast_big_dask_is_pyramid_specified(data):
    layer = Image(data, is_pyramid=False)
    assert layer.data.shape == data.shape


@pytest.mark.skip(reason="currently fails as we try and generate pyramid")
@pytest.mark.timeout(2)
@pytest.mark.parametrize('data', [data_dask, data_zarr])
def test_timing_fast_big_dask_contrast_limits_specified(data):
    layer = Image(data, contrast_limits=[0, 1])
    assert layer.data.shape == data.shape


@pytest.mark.skip(reason="currently fails as we try and generate pyramid")
@pytest.mark.timeout(2)
@pytest.mark.parametrize('data', [data_dask, data_zarr])
def test_timing_fast_big_dask_nothing_specified(data):
    layer = Image(data)
    assert layer.data.shape == data.shape


@pytest.mark.timeout(2)
def test_non_visible_images():
    """Test loading non-visible images doesn't trigger compute."""
    layer = Image(
        data_dask_2D, visible=False, is_pyramid=False, contrast_limits=[0, 1]
    )
    assert layer.data.shape == data_dask_2D.shape
