import numpy as np
import pytest

from napari.components.viewer_model import ViewerModel

dtypes = [
    np.dtype(bool),
    np.dtype(np.int8),
    np.dtype(np.uint8),
    np.dtype(np.int16),
    np.dtype(np.uint16),
    np.dtype(np.int32),
    np.dtype(np.uint32),
    np.dtype(np.int64),
    np.dtype(np.uint64),
    np.dtype(np.float16),
    np.dtype(np.float32),
    np.dtype(np.float64),
]


@pytest.mark.parametrize('dtype', dtypes)
def test_image_dytpes(dtype):
    """Test different dtype images."""
    np.random.seed(0)
    viewer = ViewerModel()

    # add dtype image data
    data = np.random.randint(20, size=(30, 40)).astype(dtype)
    viewer.add_image(data)
    assert np.all(viewer.layers[0].data == data)

    # add dtype multiscale data
    data = [
        np.random.randint(20, size=(30, 40)).astype(dtype),
        np.random.randint(20, size=(15, 20)).astype(dtype),
    ]
    viewer.add_image(data, multiscale=True)
    assert np.all(viewer.layers[1].data == data)
