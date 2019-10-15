import os
import numpy as np
from dask import array as da
from skimage.data import data_dir
from napari.util import io
import pytest


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


def test_multi_png_defaults(two_pngs):
    image_files = two_pngs
    images = io.magic_read(image_files)
    assert type(images) == da.Array
    assert images.shape == (2, 512, 512)


def test_multi_png_no_dask(two_pngs):
    image_files = two_pngs
    images = io.magic_read(image_files, use_dask=False)
    assert isinstance(images, np.ndarray)
    assert images.shape == (2, 512, 512)


def test_multi_png_no_stack(two_pngs):
    image_files = two_pngs
    images = io.magic_read(image_files, stack=False)
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
    images = io.magic_read(image_files, use_dask=False, stack=False)
    assert isinstance(images, list)
    assert len(images) == 2
    assert tuple(image.shape for image in images) == ((512, 512), (303, 384))


def test_tiff(single_tiff):
    image_files = single_tiff
    images = io.magic_read(image_files)
    assert isinstance(images, np.ndarray)
    assert images.shape == (2, 15, 10)
    assert images.dtype == np.uint8


def test_many_tiffs(single_tiff):
    image_files = single_tiff * 3
    images = io.magic_read(image_files)
    assert isinstance(images, da.Array)
    assert images.shape == (3, 2, 15, 10)
    assert images.dtype == np.uint8
