import pytest

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
    assert dims.order == [0, 1, 2, 3]
    assert dims.ndisplay == 2

    dims.order = [2, 3, 1, 0]
    assert dims.order == [2, 3, 1, 0]


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
    # On instantiation the last two dims are set to sliced mode
    assert dims.indices == (0,) * 2 + (slice(None, None, None),) * 2
    print(dims.point, dims.ndim, dims.indices)

    # Set the values of the first two dims in point mode outside of range
    dims.set_point(0, 2)
    dims.set_point(1, 3)
    assert dims.indices == (1, 1) + (slice(None, None, None),) * 2

    # Increase range and then set points again
    # Note changing the step size changes the indices for the same point value
    dims.set_range(0, (0, 4, 2))
    dims.set_range(1, (0, 4, 2))
    dims.set_point(0, 2)
    dims.set_point(1, 3)
    assert dims.indices == (1, 2) + (slice(None, None, None),) * 2


def test_axis_labels():
    dims = Dims(4)
    assert dims.axis_labels == [str(x) for x in reversed(range(4))]


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


def test_labels_order_when_changing_dims():
    dims = Dims(4)
    dims.axis_labels[-1] = 'abc'
    print(dims.axis_labels)
    dims.ndim = 5
    assert dims.axis_labels == ['4', '3', '2', '1', 'abc']


def test_labels_on_init():
    pass


@pytest.mark.parametrize(
    "ndim, ax_input, expected", [(2, 1, 1), (2, -1, 1), (4, -3, 1)]
)
def test_assert_axis_inbound(ndim, ax_input, expected):
    dims = Dims(ndim)
    actual = dims._assert_axis_inbound(ax_input)
    assert actual == expected


@pytest.mark.parametrize("ndim, ax_input", [(2, 2), (2, -3)])
def test_assert_axis_out_of_bounds(ndim, ax_input):
    dims = Dims(ndim)
    with pytest.raises(ValueError):
        dims._assert_axis_inbound(ax_input)


@pytest.mark.parametrize(
    "ndim, propname, propvalue, default, expected",
    [
        (3, 'order', range(3), [], range(3)),
        (
            3,
            'axis_labels',
            None,
            [str(ax) for ax in range(3)],
            [str(ax) for ax in range(3)],
        ),
        (2, 'order', None, [], []),
    ],
)
def test_assert_valid_init(ndim, propname, propvalue, default, expected):
    actual = Dims._assert_valid_init(ndim, propname, propvalue, default)
    assert actual == expected


@pytest.mark.parametrize(
    "ndim, propname, propvalue, default",
    [
        (3, 'order', range(4), []),
        (3, 'axis_labels', ('ax',), [str(ax) for ax in range(3)]),
        (2, 'order', [1], []),
    ],
)
def test_assert_invalid_init_raises(ndim, propname, propvalue, default):
    with pytest.raises(ValueError):
        Dims._assert_valid_init(ndim, propname, propvalue, default)
