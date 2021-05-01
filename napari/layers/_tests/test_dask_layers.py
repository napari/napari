from contextlib import contextmanager
from distutils.version import LooseVersion

import dask
import dask.array as da
import numpy as np
import pytest

from napari import layers, utils, viewer


def test_dask_array_doesnt_create_cache():
    """Test that dask arrays don't create cache but turns off fusion."""
    # by default we have no dask_cache and task fusion is active
    original = dask.config.get("optimization.fuse.active", None)

    def mock_set_view_slice():
        assert dask.config.get("optimization.fuse.active") is False

    layer = layers.Image(da.ones((100, 100)))
    layer._set_view_slice = mock_set_view_slice
    layer.set_view_slice()
    # adding a dask array won't create cache, but will turn off task fusion,
    # *but only* during slicing (see "mock_set_view_slice" above)
    assert utils.dask_cache is None
    assert dask.config.get("optimization.fuse.active", None) == original

    # if the dask version is too low to remove task fusion, emit a warning
    _dask_ver = dask.__version__
    dask.__version__ = '2.14.0'
    with pytest.warns(UserWarning) as record:
        _ = layers.Image(da.ones((100, 100)))

    assert 'upgrade Dask to v2.15.0 or later' in record[0].message.args[0]
    dask.__version__ = _dask_ver

    # make sure we can resize the cache
    utils.resize_dask_cache(10000)
    assert utils.dask_cache.cache.available_bytes == 10000

    # This should only affect dask arrays, and not numpy data
    def mock_set_view_slice2():
        assert dask.config.get("optimization.fuse.active", None) == original

    layer2 = layers.Image(np.ones((100, 100)))
    layer2._set_view_slice = mock_set_view_slice2
    layer2.set_view_slice()

    # clean up cache
    utils.dask_cache = None


def test_list_of_dask_arrays_doesnt_create_cache():
    """Test that adding a list of dask array also creates a dask cache."""
    utils.dask_cache = None  # in case other tests created it
    original = dask.config.get("optimization.fuse.active", None)
    _ = layers.Image([da.ones((100, 100)), da.ones((20, 20))])
    assert utils.dask_cache is None
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


@pytest.mark.skipif(
    dask.__version__ < LooseVersion('2.15.0'),
    reason="requires dask 2.15.0 or higher",
)
@pytest.mark.sync_only
def test_dask_optimized_slicing(delayed_dask_stack, monkeypatch):
    """Test that dask_configure reduces compute with dask stacks."""

    # make sure we have a cache
    # big enough for 10+ (10, 10, 10) "timepoints"
    utils.resize_dask_cache(100000)

    # add dask stack to the viewer, making sure to pass multiscale and clims
    v = viewer.ViewerModel()
    dask_stack = delayed_dask_stack['stack']
    v.add_image(dask_stack, multiscale=False, contrast_limits=(0, 1))
    assert delayed_dask_stack['calls'] == 1  # the first stack will be loaded

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


@pytest.mark.skipif(
    dask.__version__ < LooseVersion('2.15.0'),
    reason="requires dask 2.15.0 or higher",
)
@pytest.mark.sync_only
def test_dask_unoptimized_slicing(delayed_dask_stack, monkeypatch):
    """Prove that the dask_configure function works with a counterexample."""
    # make sure we are not caching for this test, which also tests that we
    # can turn off caching
    utils.resize_dask_cache(0)
    assert utils.dask_cache.cache.available_bytes == 0

    # mock the dask_configure function to return a no-op.
    def mock_dask_config(data):
        @contextmanager
        def dask_optimized_slicing(*args, **kwds):
            yield {}

        return dask_optimized_slicing

    monkeypatch.setattr(layers.base.base, 'configure_dask', mock_dask_config)

    # add dask stack to viewer.
    v = viewer.ViewerModel()
    dask_stack = delayed_dask_stack['stack']
    v.add_image(dask_stack, multiscale=False, contrast_limits=(0, 1))
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
    utils.resize_dask_cache(100000)

    # add dask stack to the viewer, making sure to pass multiscale and clims

    v = viewer.ViewerModel()
    dask_stack = delayed_dask_stack['stack']

    v.add_image(dask_stack, multiscale=False, contrast_limits=(0, 1))
    assert utils.dask_cache.cache.available_bytes > 0
    # make sure the cache actually has been populated
    assert len(utils.dask_cache.cache.heap.heap) > 0

    # we can resize that cache back to 0 bytes
    utils.resize_dask_cache(0)
    assert utils.dask_cache.cache.available_bytes == 0

    # adding a 2nd stack should not adjust the cache size once created
    v.add_image(dask_stack, multiscale=False, contrast_limits=(0, 1))
    assert utils.dask_cache.cache.available_bytes == 0
    # and the cache will remain empty regardless of what we do
    for i in range(3):
        v.dims.set_point(1, i)
    assert len(utils.dask_cache.cache.heap.heap) == 0

    # but we can always spin it up again
    utils.resize_dask_cache(1e4)
    assert utils.dask_cache.cache.available_bytes == 1e4
    # and adding a new image doesn't change the size
    v.add_image(dask_stack, multiscale=False, contrast_limits=(0, 1))
    assert utils.dask_cache.cache.available_bytes == 1e4
    # but the cache heap is getting populated again
    for i in range(3):
        v.dims.set_point(0, i)
    assert len(utils.dask_cache.cache.heap.heap) > 0


def test_prevent_dask_cache(delayed_dask_stack):
    """Test that pre-emptively setting cache to zero keeps it off"""
    # the del is not required, it just shows that prior state of the cache
    # does not matter... calling resize_dask_cache(0) will permanently disable
    del utils.dask_cache
    utils.resize_dask_cache(0)

    v = viewer.ViewerModel()
    dask_stack = delayed_dask_stack['stack']
    # adding a new stack will not increase the cache size
    v.add_image(dask_stack, multiscale=False, contrast_limits=(0, 1))
    assert utils.dask_cache.cache.available_bytes == 0
    # and the cache will not be populated
    for i in range(3):
        v.dims.set_point(0, i)
    assert len(utils.dask_cache.cache.heap.heap) == 0
