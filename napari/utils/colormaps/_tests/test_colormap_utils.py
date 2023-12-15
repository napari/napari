import numpy as np
import pytest

from napari.utils.colormaps.colormap_utils import label_colormap

FIRST_COLORS = [
    [0.47058824, 0.14509805, 0.02352941, 1.0],
    [0.35686275, 0.8352941, 0.972549, 1.0],
    [0.57254905, 0.5372549, 0.9098039, 1.0],
    [0.42352942, 0.00784314, 0.75686276, 1.0],
    [0.2784314, 0.22745098, 0.62352943, 1.0],
    [0.67058825, 0.9254902, 0.5411765, 1.0],
    [0.56078434, 0.6784314, 0.69803923, 1.0],
    [0.5254902, 0.5647059, 0.6039216, 1.0],
    [0.99607843, 0.96862745, 0.10980392, 1.0],
    [0.96862745, 0.26666668, 0.23137255, 1.0],
]


@pytest.mark.parametrize("index, expected", enumerate(FIRST_COLORS, start=1))
def test_label_colormap(index, expected):
    """Test the label colormap.

    Make sure that the default label colormap colors are identical
    to past versions, for UX consistency.
    """
    np.testing.assert_almost_equal(label_colormap(49).map(index), [expected])
