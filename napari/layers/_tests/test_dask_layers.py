import pytest
from napari.layers import Image
import dask.array as da
import numpy as np
import dask
from napari import utils


def test_dask_array_creates_cache():
    """Test that adding a dask array creates a dask cache and turns of fusion.
    """
    # by default we have no dask_cache and task fusion is active
    assert utils.dask_cache is None
    assert dask.config.get("optimization.fuse.active")

    def mock_set_view_slice():
        assert not dask.config.get("optimization.fuse.active")

    layer = Image(da.ones((100, 100)))
    layer._set_view_slice = mock_set_view_slice
    layer.set_view_slice()
    # adding a dask array will turn on the cache, and turn off task fusion.
    assert isinstance(utils.dask_cache, dask.cache.Cache)
    assert dask.config.get("optimization.fuse.active")

    # if the dask version is too low to remove task fusion, emit a warning
    _dask_ver = dask.__version__
    dask.__version__ = '2.14.0'
    with pytest.warns(UserWarning) as record:
        _ = Image(da.ones((100, 100)))

    assert 'upgrade Dask to v2.15.0 or later' in record[0].message.args[0]

    # make sure we can resize the cache
    assert utils.dask_cache.cache.total_bytes >= 1000
    utils.resize_dask_cache(1000)
    assert utils.dask_cache.cache.total_bytes <= 1000

    # This should only affect dask arrays, and not numpy data
    def mock_set_view_slice2():
        assert dask.config.get("optimization.fuse.active")

    layer2 = Image(np.ones((100, 100)))
    layer2._set_view_slice = mock_set_view_slice2
    layer2.set_view_slice()

    dask.__version__ = _dask_ver
    utils.dask_cache = None


def test_list_of_dask_arrays_creates_cache():
    """Test that adding a list of dask array also creates a dask cache."""
    assert utils.dask_cache is None
    assert dask.config.get("optimization.fuse.active")
    _ = Image([da.ones((100, 100)), da.ones((20, 20))])
    assert isinstance(utils.dask_cache, dask.cache.Cache)
    assert dask.config.get("optimization.fuse.active")
    # cleanup
    utils.dask_cache = None
