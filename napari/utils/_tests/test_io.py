import numpy as np
import pytest

from napari.utils.io import imsave

pytest.importorskip(
    'qtpy',
    reason='Cannot test image content without qtpy',
)


@pytest.mark.parametrize(
    "image_file", ["image", "image.png", "image.tif", "image.bmp"]
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
    from qtpy.QtGui import QImage

    from napari._qt.utils import QImg2array

    img_to_array = QImg2array(QImage(str(image_file_path)))
    assert np.equal(data, img_to_array[:, :, 0]).all()
