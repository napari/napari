from contextlib import nullcontext

import dask
import dask.array as da
import numpy as np
import pytest

from napari import layers
from napari.components import ViewerModel
from napari.utils import _dask_utils, resize_dask_cache


@pytest.mark.sync_only
def test_dask_not_greedy():
    """Make sure that we don't immediately calculate dask arrays."""

    FETCH_COUNT = 0

    def get_plane(block_id):
        if block_id:
            nonlocal FETCH_COUNT
            FETCH_COUNT += 1
        return np.random.rand(1, 1, 1, 10, 10)

    arr = da.map_blocks(
        get_plane,
        chunks=((1,) * 4, (1,) * 2, (1,) * 8, (10,), (10,)),
        dtype=float,
    )
    layer = layers.Image(arr)
    assert FETCH_COUNT == 1
    expected = (np.min(arr[0, 0, 0]), np.max(arr[0, 0, 0]))
    assert tuple(layer.contrast_limits) != expected
    FETCH_COUNT = 1  # because we just fetched one more time

    arr2 = da.map_blocks(
        get_plane,
        chunks=((1,) * 4, (1,) * 4, (1,) * 4, (10,), (10,)),
        dtype='uint8',
    )
    layer = layers.Image(arr2)
    assert FETCH_COUNT == 1
    assert tuple(layer.contrast_limits) == (0, 255)


def test_dask_array_creates_cache():
    """Test that dask arrays create cache but turns off fusion."""
    resize_dask_cache(1)
    assert _dask_utils._DASK_CACHE.cache.available_bytes == 1
    # by default we have no dask_cache and task fusion is active
    original = dask.config.get("optimization.fuse.active", None)

    def mock_set_view_slice():
        assert dask.config.get("optimization.fuse.active") is False

    layer = layers.Image(da.ones((100, 100)))
    layer._set_view_slice = mock_set_view_slice
    layer.set_view_slice()

    # adding a dask array will reate cache and turn off task fusion,
    # *but only* during slicing (see "mock_set_view_slice" above)
    assert _dask_utils._DASK_CACHE.cache.available_bytes > 100
    assert not _dask_utils._DASK_CACHE.active
    assert dask.config.get("optimization.fuse.active", None) == original

    # make sure we can resize the cache
    resize_dask_cache(10000)
    assert _dask_utils._DASK_CACHE.cache.available_bytes == 10000

    # This should only affect dask arrays, and not numpy data
    def mock_set_view_slice2():
        assert dask.config.get("optimization.fuse.active", None) == original

    layer2 = layers.Image(np.ones((100, 100)))
    layer2._set_view_slice = mock_set_view_slice2
    layer2.set_view_slice()


def test_list_of_dask_arrays_doesnt_create_cache():
    """Test that adding a list of dask array also creates a dask cache."""
    resize_dask_cache(1)  # in case other tests created it
    assert _dask_utils._DASK_CACHE.cache.available_bytes == 1
    original = dask.config.get("optimization.fuse.active", None)
    _ = layers.Image([da.ones((100, 100)), da.ones((20, 20))])
    assert _dask_utils._DASK_CACHE.cache.available_bytes > 100
    assert not _dask_utils._DASK_CACHE.active
    assert dask.config.get("optimization.fuse.active", None) == original


@pytest.fixture
def delayed_dask_stack():
    """A 4D (20, 10, 10, 10) delayed dask array, simulates disk io."""
    # we will return a dict with a 'calls' variable that tracks call count
    output = {'calls': 0}

    # create a delayed version of function that simply generates np.arrays
    # but also counts when it has been called
    @dask.delayed
    def get_array():
        nonlocal output
        output['calls'] += 1
        return np.random.rand(10, 10, 10)

    # then make a mock "timelapse" of 3D stacks
    # see https://napari.org/tutorials/applications/dask.html for details
    _list = [get_array() for fn in range(20)]
    output['stack'] = da.stack(
        [da.from_delayed(i, shape=(10, 10, 10), dtype=float) for i in _list]
    )
    assert output['stack'].shape == (20, 10, 10, 10)
    return output


@pytest.mark.sync_only
def test_dask_global_optimized_slicing(delayed_dask_stack, monkeypatch):
    """Test that dask_configure reduces compute with dask stacks."""

    # add dask stack to the viewer, making sure to pass multiscale and clims
    v = ViewerModel()
    dask_stack = delayed_dask_stack['stack']
    layer = v.add_image(dask_stack)
    assert delayed_dask_stack['calls'] == 1  # the first stack will be loaded

    with layer.dask_optimized_slicing() as (_, cache):
        assert cache.cache.available_bytes > 0
        assert cache.active
        # make sure the cache actually has been populated
        assert len(cache.cache.heap.heap) > 0

    assert not cache.active  # only active inside of the context

    # changing the Z plane should never incur calls
    # since the stack has already been loaded (& it is chunked as a 3D array)
    for i in range(3):
        v.dims.set_point(1, i)
        assert delayed_dask_stack['calls'] == 1  # still just the first call

    # changing the timepoint will, of course, incur some compute calls
    v.dims.set_point(0, 1)
    assert delayed_dask_stack['calls'] == 2
    v.dims.set_point(0, 2)
    assert delayed_dask_stack['calls'] == 3

    # but going back to previous timepoints should not, since they are cached
    v.dims.set_point(0, 1)
    v.dims.set_point(0, 0)
    assert delayed_dask_stack['calls'] == 3
    v.dims.set_point(0, 3)
    assert delayed_dask_stack['calls'] == 4


@pytest.mark.sync_only
def test_dask_unoptimized_slicing(delayed_dask_stack, monkeypatch):
    """Prove that the dask_configure function works with a counterexample."""
    # we start with a cache...but then intentionally turn it off per-layer.
    resize_dask_cache(10000)
    assert _dask_utils._DASK_CACHE.cache.available_bytes == 10000

    # add dask stack to viewer.
    v = ViewerModel()
    dask_stack = delayed_dask_stack['stack']
    layer = v.add_image(dask_stack, cache=False)
    assert delayed_dask_stack['calls'] == 1

    with layer.dask_optimized_slicing() as (_, cache):
        assert cache is None

    # without optimized dask slicing, we get a new call to the get_array func
    # (which "re-reads" the full z stack) EVERY time we change the Z plane
    # even though we've already read this full timepoint.
    for i in range(3):
        v.dims.set_point(1, i)
        assert delayed_dask_stack['calls'] == 1 + i  # 😞

    # of course we still incur calls when moving to a new timepoint...
    v.dims.set_point(0, 1)
    v.dims.set_point(0, 2)
    assert delayed_dask_stack['calls'] == 5

    # without the cache we ALSO incur calls when returning to previously loaded
    # timepoints 😭
    v.dims.set_point(0, 1)
    v.dims.set_point(0, 0)
    v.dims.set_point(0, 3)
    # all told, we have ~2x as many calls as the optimized version above.
    # (should be exactly 8 calls, but for some reason, sometimes less on CI)
    assert delayed_dask_stack['calls'] >= 7


@pytest.mark.sync_only
def test_dask_local_unoptimized_slicing(delayed_dask_stack, monkeypatch):
    """Prove that the dask_configure function works with a counterexample."""
    # make sure we are not caching for this test, which also tests that we
    # can turn off caching
    resize_dask_cache(0)
    assert _dask_utils._DASK_CACHE.cache.available_bytes == 0

    monkeypatch.setattr(
        layers.base.base, 'configure_dask', lambda *_: nullcontext
    )

    # add dask stack to viewer.
    v = ViewerModel()
    dask_stack = delayed_dask_stack['stack']
    v.add_image(dask_stack, cache=False)
    assert delayed_dask_stack['calls'] == 1

    # without optimized dask slicing, we get a new call to the get_array func
    # (which "re-reads" the full z stack) EVERY time we change the Z plane
    # even though we've already read this full timepoint.
    for i in range(3):
        v.dims.set_point(1, i)
        assert delayed_dask_stack['calls'] == 1 + i  # 😞

    # of course we still incur calls when moving to a new timepoint...
    v.dims.set_point(0, 1)
    v.dims.set_point(0, 2)
    assert delayed_dask_stack['calls'] == 5

    # without the cache we ALSO incur calls when returning to previously loaded
    # timepoints 😭
    v.dims.set_point(0, 1)
    v.dims.set_point(0, 0)
    v.dims.set_point(0, 3)
    # all told, we have ~2x as many calls as the optimized version above.
    # (should be exactly 8 calls, but for some reason, sometimes less on CI)
    assert delayed_dask_stack['calls'] >= 7


@pytest.mark.sync_only
def test_dask_cache_resizing(delayed_dask_stack):
    """Test that we can spin up, resize, and spin down the cache."""

    # make sure we have a cache
    # big enough for 10+ (10, 10, 10) "timepoints"
    resize_dask_cache(100000)

    # add dask stack to the viewer, making sure to pass multiscale and clims

    v = ViewerModel()
    dask_stack = delayed_dask_stack['stack']

    v.add_image(dask_stack)
    assert _dask_utils._DASK_CACHE.cache.available_bytes > 0
    # make sure the cache actually has been populated
    assert len(_dask_utils._DASK_CACHE.cache.heap.heap) > 0

    # we can resize that cache back to 0 bytes
    resize_dask_cache(0)
    assert _dask_utils._DASK_CACHE.cache.available_bytes == 0

    # adding a 2nd stack should not adjust the cache size once created
    v.add_image(dask_stack)
    assert _dask_utils._DASK_CACHE.cache.available_bytes == 0
    # and the cache will remain empty regardless of what we do
    for i in range(3):
        v.dims.set_point(1, i)
    assert len(_dask_utils._DASK_CACHE.cache.heap.heap) == 0

    # but we can always spin it up again
    resize_dask_cache(1e4)
    assert _dask_utils._DASK_CACHE.cache.available_bytes == 1e4
    # and adding a new image doesn't change the size
    v.add_image(dask_stack)
    assert _dask_utils._DASK_CACHE.cache.available_bytes == 1e4
    # but the cache heap is getting populated again
    for i in range(3):
        v.dims.set_point(0, i)
    assert len(_dask_utils._DASK_CACHE.cache.heap.heap) > 0


def test_prevent_dask_cache(delayed_dask_stack):
    """Test that pre-emptively setting cache to zero keeps it off"""
    resize_dask_cache(0)

    v = ViewerModel()
    dask_stack = delayed_dask_stack['stack']
    # adding a new stack will not increase the cache size
    v.add_image(dask_stack)
    assert _dask_utils._DASK_CACHE.cache.available_bytes == 0
    # and the cache will not be populated
    for i in range(3):
        v.dims.set_point(0, i)
    assert len(_dask_utils._DASK_CACHE.cache.heap.heap) == 0
