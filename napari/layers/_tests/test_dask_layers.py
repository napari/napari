from contextlib import contextmanager

import dask
import dask.array as da
import numpy as np
import pytest

from napari import layers, utils, viewer


def test_dask_array_creates_cache():
    """Test that adding a dask array creates a dask cache and turns of fusion.
    """
    # by default we have no dask_cache and task fusion is active
    assert dask.config.get("optimization.fuse.active")

    def mock_set_view_slice():
        assert not dask.config.get("optimization.fuse.active")

    layer = layers.Image(da.ones((100, 100)))
    layer._set_view_slice = mock_set_view_slice
    layer.set_view_slice()
    # adding a dask array will turn on the cache, and turn off task fusion.
    assert isinstance(utils.dask_cache, dask.cache.Cache)
    assert dask.config.get("optimization.fuse.active")

    # if the dask version is too low to remove task fusion, emit a warning
    _dask_ver = dask.__version__
    dask.__version__ = '2.14.0'
    with pytest.warns(UserWarning) as record:
        _ = layers.Image(da.ones((100, 100)))

    assert 'upgrade Dask to v2.15.0 or later' in record[0].message.args[0]
    dask.__version__ = _dask_ver

    # make sure we can resize the cache
    assert utils.dask_cache.cache.total_bytes > 1000
    utils.resize_dask_cache(1000)
    assert utils.dask_cache.cache.total_bytes <= 1000

    # This should only affect dask arrays, and not numpy data
    def mock_set_view_slice2():
        assert dask.config.get("optimization.fuse.active")

    layer2 = layers.Image(np.ones((100, 100)))
    layer2._set_view_slice = mock_set_view_slice2
    layer2.set_view_slice()


def test_list_of_dask_arrays_creates_cache():
    """Test that adding a list of dask array also creates a dask cache."""
    assert dask.config.get("optimization.fuse.active")
    _ = layers.Image([da.ones((100, 100)), da.ones((20, 20))])
    assert isinstance(utils.dask_cache, dask.cache.Cache)
    assert dask.config.get("optimization.fuse.active")


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
        [da.from_delayed(i, shape=(10, 10, 10), dtype=np.float) for i in _list]
    )
    assert output['stack'].shape == (20, 10, 10, 10)
    return output


def test_dask_optimized_slicing(delayed_dask_stack, monkeypatch):
    """Test that dask_configure reduces compute with dask stacks."""

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


def test_dask_unoptimized_slicing(delayed_dask_stack, monkeypatch):
    """Prove that the dask_configure function works with a counterexample."""
    # make sure we are not caching for this test, which also tests that we
    # can turn off caching
    utils.resize_dask_cache(0)

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
    # all told, we have 2x as many calls as the optimized version above.
    assert delayed_dask_stack['calls'] == 8
