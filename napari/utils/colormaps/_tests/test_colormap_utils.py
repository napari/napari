import numpy as np
import pytest

from napari.utils.colormaps.colormap_utils import label_colormap

FIRST_COLORS = [
    [0.47063142, 0.14611654, 0.027308635, 1.0],
    [0.35923997, 0.83787304, 0.9764158, 1.0],
    [0.57314, 0.53869504, 0.9130728, 1.0],
    [0.42733493, 0.009019371, 0.75742406, 1.0],
    [0.28053862, 0.22821146, 0.6264092, 1.0],
    [0.67241573, 0.92709625, 0.5439105, 1.0],
    [0.5636559, 0.68220073, 0.7002792, 1.0],
    [0.5277779, 0.5672113, 0.6043446, 1.0],
    [0.9987752, 0.9686924, 0.10985588, 1.0],
    [0.97181, 0.27003965, 0.23497851, 1.0],
]


@pytest.mark.parametrize("index, expected", enumerate(FIRST_COLORS, start=1))
def test_label_colormap(index, expected):
    """Test the label colormap.

    Make sure that the default label colormap colors are identical
    to past versions, for UX consistency.
    """
    np.testing.assert_almost_equal(label_colormap(49).map(index), [expected])
