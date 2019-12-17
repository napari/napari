import os
from pathlib import Path

import numpy as np
from dask import array as da
from skimage.data import data_dir
from tempfile import TemporaryDirectory
from napari.utils import io
import pytest


try:
    import zarr

    zarr_available = True
except ImportError:
    zarr_available = False


@pytest.fixture
def single_png():
    image_files = [os.path.join(data_dir, fn) for fn in ['camera.png']]
    return image_files


@pytest.fixture
def two_pngs():
    image_files = [
        os.path.join(data_dir, fn) for fn in ['moon.png', 'camera.png']
    ]
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


def test_single_png_defaults(single_png):
    image_files = single_png
    images = io.magic_imread(image_files)
    assert isinstance(images, np.ndarray)
    assert images.shape == (512, 512)


def test_single_png_single_file(single_png):
    image_files = single_png[0]
    images = io.magic_imread(image_files)
    assert isinstance(images, np.ndarray)
    assert images.shape == (512, 512)


def test_single_png_pathlib(single_png):
    image_files = Path(single_png[0])
    images = io.magic_imread(image_files)
    assert isinstance(images, np.ndarray)
    assert images.shape == (512, 512)


def test_multi_png_defaults(two_pngs):
    image_files = two_pngs
    images = io.magic_imread(image_files)
    assert isinstance(images, da.Array)
    assert images.shape == (2, 512, 512)


def test_multi_png_pathlib(two_pngs):
    image_files = [Path(png) for png in two_pngs]
    images = io.magic_imread(image_files)
    assert isinstance(images, da.Array)
    assert images.shape == (2, 512, 512)


def test_multi_png_no_dask(two_pngs):
    image_files = two_pngs
    images = io.magic_imread(image_files, use_dask=False)
    assert isinstance(images, np.ndarray)
    assert images.shape == (2, 512, 512)


def test_multi_png_no_stack(two_pngs):
    image_files = two_pngs
    images = io.magic_imread(image_files, stack=False)
    assert isinstance(images, list)
    assert len(images) == 2
    assert all(a.shape == (512, 512) for a in images)


def test_irregular_images(irregular_images):
    image_files = irregular_images
    # Ideally, this would work "magically" with dask and irregular images,
    # but there is no foolproof way to do this without reading in all the
    # files. We need to be able to inspect the file shape without reading
    # it in first, then we can automatically turn stacking off when shapes
    # are irregular (and create proper dask arrays)
    images = io.magic_imread(image_files, use_dask=False, stack=False)
    assert isinstance(images, list)
    assert len(images) == 2
    assert tuple(image.shape for image in images) == ((512, 512), (303, 384))


def test_tiff(single_tiff):
    image_files = single_tiff
    images = io.magic_imread(image_files)
    assert isinstance(images, np.ndarray)
    assert images.shape == (2, 15, 10)
    assert images.dtype == np.uint8


def test_many_tiffs(single_tiff):
    image_files = single_tiff * 3
    images = io.magic_imread(image_files)
    assert isinstance(images, da.Array)
    assert images.shape == (3, 2, 15, 10)
    assert images.dtype == np.uint8


def test_single_filename(single_tiff):
    image_files = single_tiff[0]
    images = io.magic_imread(image_files)
    assert images.shape == (2, 15, 10)


@pytest.mark.skipif(not zarr_available, reason='zarr not installed')
def test_zarr():
    image = np.random.random((10, 20, 20))
    with TemporaryDirectory(suffix='.zarr') as fout:
        z = zarr.open(fout, 'a', shape=image.shape)
        z[:] = image
        image_in = io.magic_imread([fout])
        # Note: due to lazy loading, the next line needs to happen within
        # the context manager. Alternatively, we could convert to NumPy here.
        np.testing.assert_array_equal(image, image_in)


@pytest.mark.skipif(not zarr_available, reason='zarr not installed')
def test_zarr_pyramid():
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
        pyramid_in = io.magic_imread([fout])
        assert len(pyramid) == len(pyramid_in)
        # Note: due to lazy loading, the next line needs to happen within
        # the context manager. Alternatively, we could convert to NumPy here.
        for images, images_in in zip(pyramid, pyramid_in):
            np.testing.assert_array_equal(images, images_in)
