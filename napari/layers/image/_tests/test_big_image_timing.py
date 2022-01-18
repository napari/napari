import time

import dask.array as da
import pytest
import zarr

from napari.layers import Image

data_dask = da.random.random(
    size=(100_000, 1000, 1000), chunks=(1, 1000, 1000)
)
data_zarr = zarr.zeros((100_000, 1000, 1000))


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(multiscale=False, contrast_limits=[0, 1]),
        dict(multiscale=False),
        dict(contrast_limits=[0, 1]),
        {},
    ],
    ids=('all', 'multiscale', 'clims', 'nothing'),
)
@pytest.mark.parametrize('data', [data_dask, data_zarr], ids=('dask', 'zarrs'))
def test_timing_fast_big_dask(data, kwargs):
    now = time.monotonic()
    assert Image(data, **kwargs).data.shape == data.shape
    elapsed = time.monotonic() - now
    assert (
        elapsed < 2
    ), "Test took to long some computation are likely not lazy"


def test_non_visible_images():
    """Test loading non-visible images doesn't trigger compute."""
    data_dask_2D = da.random.random((100_000, 100_000))
    layer = Image(
        data_dask_2D,
        visible=False,
        multiscale=False,
        contrast_limits=[0, 1],
    )
    assert layer.data.shape == data_dask_2D.shape
