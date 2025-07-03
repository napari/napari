import struct

import numpy as np
import pytest
import tifffile
from imageio.v3 import imread

from napari.utils.io import imsave


@pytest.mark.slow
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
    """Test saving a bigtiff file.

    In napari IO, we use compression when saving to tiff. bigtiff mode
    is required when data size *after compression* is over 4GB. However,
    bigtiff is not as broadly implemented as normal tiff, so we don't want to
    always use that flag. Therefore, in our internal code, we catch the error
    raised when trying to write a too-large TIFF file, then rewrite using
    bigtiff. This test checks that the mechanism works correctly.

    Generating 4GB+ of uncompressible data is expensive, so here we:

    1. Generate a smaller amount of "random" data using np.empty.
    2. Monkeypatch tifffile's save routine to raise an error *as if* bigtiff
       was required for this small dataset.
    3. This triggers saving as bigtiff.
    4. We check that the file was correctly saved as bigtiff.
    """
    old_write = tifffile.imwrite

    def raise_no_bigtiff(*args, **kwargs):
        if 'bigtiff' not in kwargs:
            raise struct.error
        old_write(*args, **kwargs)

    monkeypatch.setattr(tifffile, 'imwrite', raise_no_bigtiff)

    # create image data. It can be <4GB compressed because we
    # monkeypatched tifffile.imwrite to raise an error if bigtiff is not set
    data = np.empty((20, 200, 200), dtype='uint16')

    # create image and assert image file creation
    image_path = str(tmp_path / 'data.tif')
    imsave(image_path, data)
    with tifffile.TiffFile(image_path) as tiff:
        assert tiff.is_bigtiff
