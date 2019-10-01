import numpy as np
from napari import Viewer
import dask.array as da
import zarr
import xarray as xr


def test_dask_2D(qtbot):
    """Test adding 2D dask image."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    da.random.seed(0)
    data = da.random.random((10, 15))
    viewer.add_image(data)
    assert np.all(viewer.layers[0].data == data)

    # Close the viewer
    viewer.window.close()


def test_dask_nD(qtbot):
    """Test adding nD dask image."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    da.random.seed(0)
    data = da.random.random((10, 15, 6, 16))
    viewer.add_image(data)
    assert np.all(viewer.layers[0].data == data)

    # Close the viewer
    viewer.window.close()


def test_zarr_2D(qtbot):
    """Test adding 2D zarr image."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    data = zarr.zeros((200, 100), chunks=(40, 20))
    data[53:63, 10:20] = 1
    # If passing a zarr file directly, must pass contrast_limits
    viewer.add_image(data, contrast_limits=[0, 1])
    assert np.all(viewer.layers[0].data == data)

    # Close the viewer
    viewer.window.close()


def test_zarr_nD(qtbot):
    """Test adding nD zarr image."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    data = zarr.zeros((200, 100, 50), chunks=(40, 20, 10))
    data[53:63, 10:20, :] = 1
    # If passing a zarr file directly, must pass contrast_limits
    viewer.add_image(data, contrast_limits=[0, 1])
    assert np.all(viewer.layers[0].data == data)

    # Close the viewer
    viewer.window.close()


def test_zarr_dask_2D(qtbot):
    """Test adding 2D dask image."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    data = zarr.zeros((200, 100), chunks=(40, 20))
    data[53:63, 10:20] = 1
    zdata = da.from_zarr(data)
    viewer.add_image(zdata)
    assert np.all(viewer.layers[0].data == zdata)

    # Close the viewer
    viewer.window.close()


def test_zarr_dask_nD(qtbot):
    """Test adding nD zarr image."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    data = zarr.zeros((200, 100, 50), chunks=(40, 20, 10))
    data[53:63, 10:20, :] = 1
    zdata = da.from_zarr(data)
    viewer.add_image(zdata)
    assert np.all(viewer.layers[0].data == zdata)

    # Close the viewer
    viewer.window.close()


def test_xarray_2D(qtbot):
    """Test adding 2D xarray image."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.random((10, 15))
    xdata = xr.DataArray(data, dims=['y', 'x'])
    viewer.add_image(data)
    assert np.all(viewer.layers[0].data == xdata)

    # Close the viewer
    viewer.window.close()


def test_xarray_nD(qtbot):
    """Test adding nD xarray image."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.random((10, 15, 6, 16))
    xdata = xr.DataArray(data, dims=['t', 'z', 'y', 'x'])
    viewer.add_image(xdata)
    assert np.all(viewer.layers[0].data == xdata)

    # Close the viewer
    viewer.window.close()
