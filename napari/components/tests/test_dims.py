from napari.components import Dims
from napari.components.dims_constants import DimsMode


def test_ndim():
    """
    Test number of dimensions including after adding and removing dimensions.
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
    Test display setting.
    """
    dims = Dims(4)
    assert dims.display == [False] * 4

    dims.set_display(0, True)
    dims.set_display(1, True)
    assert dims.display == [True, True, False, False]

    dims._set_2d_viewing()
    assert dims.display == [False, False, True, True]


def test_point():
    """
    Test point setting.
    """
    dims = Dims(4)
    assert dims.point == [0] * 4

    dims.set_point(3, 4)
    assert dims.point == [0, 0, 0, 4]

    dims.set_point(2, 1)
    assert dims.point == [0, 0, 1, 4]


def test_mode():
    """
    Test mode setting.
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
    assert dims.range == [(0, 2, 1)] * 4

    dims.set_range(3, (0, 4, 2))
    assert dims.range == [(0, 2, 1)] * 3 + [(0, 4, 2)]


def test_interval():
    """
    Test interval setting.
    """
    dims = Dims(4)
    assert dims.interval == [(0, 1)] * 4

    dims.set_interval(3, (0, 3))
    assert dims.interval == [(0, 1)] * 3 + [(0, 3)]


def test_indices():
    """
    Test indices values.
    """
    dims = Dims(4)
    # On instantiation no dims are displayed and the indices default to 0
    assert dims.indices == (0,) * 4

    dims._set_2d_viewing()
    # On 2D viewing the last two dims are now set to sliced mode
    assert dims.indices == (0,) * 2 + (slice(None, None, None),) * 2

    dims.set_point(0, 2)
    dims.set_point(1, 3)
    # Set the values of the first two dims in point mode
    assert dims.indices == (2, 3) + (slice(None, None, None),) * 2


def test_order_when_changing_ndim():
    """
    Test order of the dims when changing the number of dimensions.
    """
    dims = Dims(4)
    dims.set_point(0, 2)

    dims.ndim = 5
    # Test that new dims get appended to the beginning of lists
    assert dims.point == [0, 2, 0, 0, 0]

    dims.set_point(2, 3)
    dims.ndim = 3
    # Test that dims get removed from the beginning of lists
    assert dims.point == [3, 0, 0]
