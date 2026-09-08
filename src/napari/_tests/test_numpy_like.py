import dask.array as da
import numpy as np
import xarray as xr
import zarr
from numpy.testing import assert_array_equal

from napari.components.viewer_model import ViewerModel


def test_dask_2D():
    """Test adding 2D dask image."""
    viewer = ViewerModel()

    da.random.seed(0)
    data = da.random.random((10, 15))
    viewer.add_image(data)
    assert_array_equal(viewer.layers[0].data, data)


def test_dask_nD():
    """Test adding nD dask image."""
    viewer = ViewerModel()

    da.random.seed(0)
    data = da.random.random((10, 15, 6, 16))
    viewer.add_image(data)
    assert_array_equal(viewer.layers[0].data, data)


def test_zarr_2D():
    """Test adding 2D zarr image."""
    viewer = ViewerModel()

    data = zarr.zeros((200, 100), chunks=(40, 20))
    data[53:63, 10:20] = 1
    # If passing a zarr file directly, must pass contrast_limits
    viewer.add_image(data, contrast_limits=[0, 1])
    assert_array_equal(viewer.layers[0].data, data)


def test_zarr_nD():
    """Test adding nD zarr image."""
    viewer = ViewerModel()

    data = zarr.zeros((200, 100, 50), chunks=(40, 20, 10))
    data[53:63, 10:20, :] = 1
    # If passing a zarr file directly, must pass contrast_limits
    viewer.add_image(data, contrast_limits=[0, 1])
    assert_array_equal(viewer.layers[0].data, data)


def test_zarr_dask_2D():
    """Test adding 2D dask image."""
    viewer = ViewerModel()

    data = zarr.zeros((200, 100), chunks=(40, 20))
    data[53:63, 10:20] = 1
    zdata = da.from_zarr(data)
    viewer.add_image(zdata)
    assert_array_equal(viewer.layers[0].data, zdata)


def test_zarr_dask_nD():
    """Test adding nD zarr image."""
    viewer = ViewerModel()

    data = zarr.zeros((200, 100, 50), chunks=(40, 20, 10))
    data[53:63, 10:20, :] = 1
    zdata = da.from_zarr(data)
    viewer.add_image(zdata)
    assert_array_equal(viewer.layers[0].data, zdata)


def test_xarray_2D():
    """Test adding 2D xarray image."""
    viewer = ViewerModel()

    np.random.seed(0)
    data = np.random.random((10, 15))
    xdata = xr.DataArray(data, dims=['y', 'x'])
    viewer.add_image(data)
    assert_array_equal(viewer.layers[0].data, xdata)


def test_xarray_nD():
    """Test adding nD xarray image."""
    viewer = ViewerModel()

    np.random.seed(0)
    data = np.random.random((10, 15, 6, 16))
    xdata = xr.DataArray(data, dims=['t', 'z', 'y', 'x'])
    viewer.add_image(xdata)
    assert_array_equal(viewer.layers[0].data, xdata)
