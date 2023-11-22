import numpy as np
import pytest

from napari._vispy.utils.gl import fix_data_dtype

BUILTINS_DISP = 'napari'
BUILTINS_NAME = 'builtins'


# Previously tests often segfaulted on CI at the 26th test of test_qt_viewer
# That test (number 26) was split off to make debugging easier
# See https://github.com/napari/napari/pull/5676
@pytest.mark.parametrize(
    "dtype",
    [
        'int8',
        'uint8',
        'int16',
        'uint16',
        'int32',
        'float16',
        'float32',
        'float64',
    ],
)
def test_qt_viewer_data_integrity(make_napari_viewer, dtype):
    """Test that the viewer doesn't change the underlying array."""
    image = np.random.rand(10, 32, 32)
    image *= 200 if dtype.endswith('8') else 2**14
    image = image.astype(dtype)
    imean = image.mean()

    viewer = make_napari_viewer()
    layer = viewer.add_image(image.copy())
    data = layer.data

    datamean = np.mean(data)
    assert datamean == imean
    # toggle dimensions
    viewer.dims.ndisplay = 3
    datamean = np.mean(data)
    assert datamean == imean
    # back to 2D
    viewer.dims.ndisplay = 2
    datamean = np.mean(data)
    assert datamean == imean
    # also check that vispy gets (almost) the same data
    datamean = np.mean(fix_data_dtype(data))
    assert np.allclose(datamean, imean, rtol=5e-04)


def test_fix_data_dtype_big_values():
    data = np.array([0, 2, 2**17], dtype=np.uint32)
    casted = fix_data_dtype(data)
    assert np.allclose(casted, data)
    assert casted.dtype == np.float32
