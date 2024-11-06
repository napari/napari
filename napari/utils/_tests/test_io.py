import struct

import numpy as np
import pytest
import tifffile
from imageio.v3 import imread

from napari.utils.io import imsave


@pytest.mark.parametrize(
    'image_file', ['image', 'image.png', 'image.tif', 'image.bmp']
)
def test_imsave(tmp_path, image_file):
    """Save data to image file."""
    # create image data
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15), dtype=np.ubyte)
    image_file_path = tmp_path / image_file
    assert not image_file_path.is_file()

    # create image and assert image file creation
    imsave(str(image_file_path), data)
    assert image_file_path.is_file()

    # check image content
    img_to_array = imread(str(image_file_path))
    assert np.equal(data, img_to_array).all()


def test_imsave_bool_tiff(tmp_path):
    """Test saving a boolean array to a tiff file."""
    np.random.seed(0)
    data = np.random.randint(low=0, high=2, size=(10, 15), dtype=bool)
    image_file_path = tmp_path / 'bool_image.tif'
    assert not image_file_path.is_file()

    # create image and assert image file creation
    imsave(str(image_file_path), data)
    assert image_file_path.is_file()

    # check image content
    img_to_array = imread(str(image_file_path))
    assert np.equal(data, img_to_array).all()


@pytest.mark.parametrize(
    'image_file', ['image', 'image.png', 'image.tif', 'image.bmp']
)
def test_imsave_float(tmp_path, image_file):
    """Test saving float image data."""
    # create image data
    np.random.seed(0)
    data = np.random.random((10, 15))
    image_file_path = tmp_path / image_file
    assert not image_file_path.is_file()

    # create image
    imsave(str(image_file_path), data)
    # only TIF can store float
    if image_file.endswith('.tif'):
        assert image_file_path.is_file()
        img_to_array = imread(str(image_file_path))
        assert np.equal(data, img_to_array).all()
    # if no EXT provided, for float data should save as .tif
    elif image_file == 'image':
        assert image_file_path.with_suffix('.tif').is_file()
        img_to_array = imread(str(image_file_path.with_suffix('.tif')))
        assert np.equal(data, img_to_array).all()

    else:
        assert not image_file_path.is_file()


def test_imsave_large_file(monkeypatch, tmp_path):
    old_write = tifffile.imwrite

    def raise_no_bigtiff(*args, **kwargs):
        if 'bigtiff' not in kwargs:
            raise struct.error
        old_write(*args, **kwargs)

    monkeypatch.setattr(tifffile, 'imwrite', raise_no_bigtiff)

    data = np.random.randint(
        low=0, high=2**16, size=(128, 4096, 4096), dtype='uint16'
    )  # 4GB size

    # create image and assert image file creation
    image_path = str(tmp_path / 'data.tif')
    imsave(image_path, data)
    with tifffile.TiffFile(image_path) as tiff:
        assert tiff.is_bigtiff
