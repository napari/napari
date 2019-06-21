from napari.components import Dims
from napari.components._dims._constants import DimsMode


def test_ndim():
    """
    Tests number of dimensions including after adding and removing dimensions.
    """
    dims = Dims()
    assert dims.ndim == 0

    dims = Dims(4)
    assert dims.ndim == 4

    dims = Dims(2)
    assert dims.ndim == 2

    dims.ndim = 10
    assert dims.ndim == 10

    dims.ndim = 5
    assert dims.ndim == 5


def test_display():
    """
    Tests display setting.
    """
    dims = Dims(4)
    assert dims.display == [False] * 4

    dims.set_display(0, True)
    dims.set_display(1, True)
    assert dims.display == [True, True, False, False]
    assert dims.displayed == [0, 1]

    dims._set_2d_viewing()
    assert dims.display == [False, False, True, True]


def test_point():
    """
    Tests point setting.
    """
    dims = Dims(4)
    assert dims.point == [0.0] * 4

    dims.set_point(3, 2.5)
    assert dims.point == [0.0, 0.0, 0.0, 2.5]

    dims.set_point(2, 0.5)
    assert dims.point == [0.0, 0.0, 0.5, 2.5]


def test_mode():
    """
    Tests mode setting.
    """
    dims = Dims(4)
    assert dims.mode == [DimsMode.POINT] * 4

    dims.set_mode(3, DimsMode.INTERVAL)
    assert dims.mode == [DimsMode.POINT] * 3 + [DimsMode.INTERVAL]


def test_range():
    """
    Tests range setting.
    """
    dims = Dims(4)
    assert dims.range == [(0.0, 1.0, 0.01)] * 4

    dims.set_range(3, (0.0, 2.0, 0.5))
    assert dims.range == [(0.0, 1.0, 0.01)] * 3 + [(0.0, 2.0, 0.5)]


def test_interval():
    """
    Tests interval setting.
    """
    dims = Dims(4)
    assert dims.interval == [(0.3, 0.7)] * 4

    dims.set_interval(3, (0.2, 0.8))
    assert dims.interval == [(0.3, 0.7)] * 3 + [(0.2, 0.8)]


def test_indices():
    """
    Tests indices values.
    """
    dims = Dims(4)
    # On instantiation no dims are displayed and the indices default to 0
    assert dims.indices == (0.0,) * 4

    dims._set_2d_viewing()
    # On 2D viewing the last two dims are now set to sliced mode
    assert dims.indices == (0.0,) * 2 + (slice(None, None, None),) * 2

    dims.set_point(0, 2)
    dims.set_point(1, 3)
    # Set the values of the first two dims in point mode
    assert dims.indices == (2, 3) + (slice(None, None, None),) * 2
