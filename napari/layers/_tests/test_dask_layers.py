import pytest
from napari.layers import Image
import dask.array as da
import dask
from napari import utils


def test_dask_array_creates_cache():
    """Test that adding a dask array creates a dask cache and turns of fusion.
    """
    # by default we have no dask_cache and task fusion is active
    utils.dask_cache = None
    assert dask.config.get("optimization.fuse.active")
    _ = Image(da.ones((100, 100)))
    # adding a dask array will turn on the cache, and turn off task fusion.
    assert isinstance(utils.dask_cache, dask.cache.Cache)
    assert not dask.config.get("optimization.fuse.active")

    # if the dask version is too low to take remove task fusion, emit a warning
    _dask_ver = dask.__version__
    dask.__version__ = '2.14.0'
    with pytest.warns(UserWarning) as record:
        _ = Image(da.ones((100, 100)))

    assert 'upgrade Dask to v2.15.0 or later' in record[0].message.args[0]

    # make sure we can resize the cache
    assert utils.dask_cache.cache.total_bytes >= 1000
    utils.resize_dask_cache(1000)
    assert utils.dask_cache.cache.total_bytes <= 1000

    # cleanup
    dask.config.set({"optimization.fuse.active": True})
    utils.dask_cache = None
    dask.__version__ = _dask_ver


def test_list_of_dask_arrays_creates_cache():
    """Test that adding a list of dask array also creates a dask cache."""
    assert dask.config.get("optimization.fuse.active")
    _ = Image([da.ones((100, 100)), da.ones((20, 20))])
    # adding a dask array will turn on the cache, and turn off task fusion.
    assert isinstance(utils.dask_cache, dask.cache.Cache)
    assert not dask.config.get("optimization.fuse.active")
