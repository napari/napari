import os
import numpy as np
from dask import array as da
from skimage.data import data_dir
from tempfile import TemporaryDirectory
import pytest
from napari.components import ViewerModel


try:
    import zarr

    zarr_available = True
except ImportError:
    zarr_available = False


@pytest.fixture
def two_pngs():
    image_files = [
        os.path.join(data_dir, fn) for fn in ['moon.png', 'camera.png']
    ]
    return image_files


@pytest.fixture
def rgb_png():
    image_files = [os.path.join(data_dir, fn) for fn in ['astronaut.png']]
    return image_files


@pytest.fixture
def single_png():
    image_files = [os.path.join(data_dir, fn) for fn in ['camera.png']]
    return image_files


@pytest.fixture
def irregular_images():
    image_files = [
        os.path.join(data_dir, fn) for fn in ['camera.png', 'coins.png']
    ]
    return image_files


@pytest.fixture
def single_tiff():
    image_files = [os.path.join(data_dir, 'multipage.tif')]
    return image_files


def test_add_single_png_defaults(single_png):
    image_files = single_png
    viewer = ViewerModel()
    viewer.add_image(path=image_files)
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 2
    assert isinstance(viewer.layers[0].data, np.ndarray)
    assert viewer.layers[0].data.shape == (512, 512)


def test_add_multi_png_defaults(two_pngs):
    image_files = two_pngs
    viewer = ViewerModel()
    viewer.add_image(path=image_files)
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 3
    assert isinstance(viewer.layers[0].data, da.Array)
    assert viewer.layers[0].data.shape == (2, 512, 512)


def test_add_tiff(single_tiff):
    image_files = single_tiff
    viewer = ViewerModel()
    viewer.add_image(path=image_files)
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 3
    assert isinstance(viewer.layers[0].data, np.ndarray)
    assert viewer.layers[0].data.shape == (2, 15, 10)
    assert viewer.layers[0].data.dtype == np.uint8


def test_add_many_tiffs(single_tiff):
    image_files = single_tiff * 3
    viewer = ViewerModel()
    viewer.add_image(path=image_files)
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 4
    assert isinstance(viewer.layers[0].data, da.Array)
    assert viewer.layers[0].data.shape == (3, 2, 15, 10)
    assert viewer.layers[0].data.dtype == np.uint8


def test_add_single_filename(single_tiff):
    image_files = single_tiff[0]
    viewer = ViewerModel()
    viewer.add_image(path=image_files)
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
        viewer.add_image(path=[fout])
        assert len(viewer.layers) == 1
        # Note: due to lazy loading, the next line needs to happen within
        # the context manager. Alternatively, we could convert to NumPy here.
        np.testing.assert_array_equal(image, viewer.layers[0].data)


@pytest.mark.skipif(not zarr_available, reason='zarr not installed')
def test_zarr_pyramid():
    viewer = ViewerModel()
    pyramid = [
        np.random.random((20, 20)),
        np.random.random((10, 10)),
        np.random.random((5, 5)),
    ]
    with TemporaryDirectory(suffix='.zarr') as fout:
        root = zarr.open_group(fout, 'a')
        for i in range(len(pyramid)):
            shape = 20 // 2 ** i
            z = root.create_dataset(str(i), shape=(shape,) * 2)
            z[:] = pyramid[i]
        viewer.add_image(path=[fout], is_pyramid=True)
        assert len(viewer.layers) == 1
        assert len(pyramid) == len(viewer.layers[0].data)
        # Note: due to lazy loading, the next line needs to happen within
        # the context manager. Alternatively, we could convert to NumPy here.
        for images, images_in in zip(pyramid, viewer.layers[0].data):
            np.testing.assert_array_equal(images, images_in)


def test_add_multichannel_rgb(rgb_png):
    image_files = rgb_png
    viewer = ViewerModel()
    viewer.add_image(path=image_files, channel_axis=2)
    assert len(viewer.layers) == 3
    assert viewer.dims.ndim == 2
    assert isinstance(viewer.layers[0].data, np.ndarray)
    assert viewer.layers[0].data.shape == (512, 512)


def test_add_multichannel_tiff(single_tiff):
    image_files = single_tiff
    viewer = ViewerModel()
    viewer.add_image(path=image_files, channel_axis=0)
    assert len(viewer.layers) == 2
    assert viewer.dims.ndim == 2
    assert isinstance(viewer.layers[0].data, np.ndarray)
    assert viewer.layers[0].data.shape == (15, 10)
    assert viewer.layers[0].data.dtype == np.uint8
