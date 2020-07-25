from tempfile import TemporaryDirectory

import numpy as np
import pytest
from dask import array as da

from napari.components import ViewerModel

try:
    import zarr

    zarr_available = True
except ImportError:
    zarr_available = False


# the following fixtures are defined in napari/conftest.py
# single_png, two_pngs, irregular_images, single_tiff, rgb_png


def test_add_single_png_defaults(single_png):
    image_files = single_png
    viewer = ViewerModel()
    viewer.open(image_files, plugin='builtins')
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 2
    assert isinstance(viewer.layers[0].data, np.ndarray)
    assert viewer.layers[0].data.shape == (512, 512)


def test_add_multi_png_defaults(two_pngs):
    image_files = two_pngs
    viewer = ViewerModel()
    viewer.open(image_files, stack=True, plugin='builtins')
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 3
    assert isinstance(viewer.layers[0].data, da.Array)
    assert viewer.layers[0].data.shape == (2, 512, 512)

    viewer.open(image_files, stack=False, plugin='builtins')
    assert len(viewer.layers) == 3


def test_add_tiff(single_tiff):
    image_files = single_tiff
    viewer = ViewerModel()
    viewer.open(image_files, plugin='builtins')
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 3
    assert isinstance(viewer.layers[0].data, np.ndarray)
    assert viewer.layers[0].data.shape == (2, 15, 10)
    assert viewer.layers[0].data.dtype == np.uint8


def test_add_many_tiffs(single_tiff):
    image_files = single_tiff * 3
    viewer = ViewerModel()
    viewer.open(image_files, stack=True, plugin='builtins')
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 4
    assert isinstance(viewer.layers[0].data, da.Array)
    assert viewer.layers[0].data.shape == (3, 2, 15, 10)
    assert viewer.layers[0].data.dtype == np.uint8


def test_add_single_filename(single_tiff):
    image_files = single_tiff[0]
    viewer = ViewerModel()
    viewer.open(image_files, plugin='builtins')
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 3
    assert isinstance(viewer.layers[0].data, np.ndarray)
    assert viewer.layers[0].data.shape == (2, 15, 10)
    assert viewer.layers[0].data.dtype == np.uint8


@pytest.mark.skipif(not zarr_available, reason='zarr not installed')
def test_add_zarr():
    viewer = ViewerModel()
    image = np.random.random((10, 20, 20))
    with TemporaryDirectory(suffix='.zarr') as fout:
        z = zarr.open(fout, 'a', shape=image.shape)
        z[:] = image
        viewer.open([fout], plugin='builtins')
        assert len(viewer.layers) == 1
        # Note: due to lazy loading, the next line needs to happen within
        # the context manager. Alternatively, we could convert to NumPy here.
        np.testing.assert_array_equal(image, viewer.layers[0].data)


@pytest.mark.skipif(not zarr_available, reason='zarr not installed')
def test_zarr_multiscale():
    viewer = ViewerModel()
    multiscale = [
        np.random.random((20, 20)),
        np.random.random((10, 10)),
        np.random.random((5, 5)),
    ]
    with TemporaryDirectory(suffix='.zarr') as fout:
        root = zarr.open_group(fout, 'a')
        for i in range(len(multiscale)):
            shape = 20 // 2 ** i
            z = root.create_dataset(str(i), shape=(shape,) * 2)
            z[:] = multiscale[i]
        viewer.open(fout, multiscale=True, plugin='builtins')
        assert len(viewer.layers) == 1
        assert len(multiscale) == len(viewer.layers[0].data)
        # Note: due to lazy loading, the next line needs to happen within
        # the context manager. Alternatively, we could convert to NumPy here.
        for images, images_in in zip(multiscale, viewer.layers[0].data):
            np.testing.assert_array_equal(images, images_in)


def test_add_multichannel_rgb(rgb_png):
    image_files = rgb_png
    viewer = ViewerModel()
    viewer.open(image_files, channel_axis=2, plugin='builtins')
    assert len(viewer.layers) == 3
    assert viewer.dims.ndim == 2
    assert isinstance(viewer.layers[0].data, np.ndarray)
    assert viewer.layers[0].data.shape == (512, 512)


def test_add_multichannel_tiff(single_tiff):
    image_files = single_tiff
    viewer = ViewerModel()
    viewer.open(image_files, channel_axis=0, plugin='builtins')
    assert len(viewer.layers) == 2
    assert viewer.dims.ndim == 2
    assert isinstance(viewer.layers[0].data, np.ndarray)
    assert viewer.layers[0].data.shape == (15, 10)
    assert viewer.layers[0].data.dtype == np.uint8
