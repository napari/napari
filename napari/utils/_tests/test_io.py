import numpy as np
import pytest

from napari.utils.io import imsave


@pytest.mark.parametrize(
    "image_file", ["image", "image.png", "image.tif", "image.jpg"]
)
def test_imsave(tmp_path, image_file):
    """Save data to image file."""
    # create image data
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15), dtype=np.ubyte)
    image_dir_path = tmp_path / "screenshots"
    image_dir_path.mkdir()
    image_file_path = image_dir_path / image_file
    assert not image_file_path.is_file()

    # create image and assert image file creation
    imsave(str(image_file_path), data)
    assert image_file_path.is_file()
