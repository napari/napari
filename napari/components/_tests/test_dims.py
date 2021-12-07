import pytest

from napari.components import Dims
from napari.components.dims import assert_axis_in_bounds


def test_ndim():
    """
    Test number of dimensions including after adding and removing dimensions.
    """
    dims = Dims()
    assert dims.ndim == 2

    dims = Dims(ndim=4)
    assert dims.ndim == 4

    dims = Dims(ndim=2)
    assert dims.ndim == 2

    dims.ndim = 10
    assert dims.ndim == 10

    dims.ndim = 5
    assert dims.ndim == 5


def test_display():
    """
    Test display setting.
    """
    dims = Dims(ndim=4)
    assert dims.order == (0, 1, 2, 3)
    assert dims.ndisplay == 2

    dims.order = (2, 3, 1, 0)
    assert dims.order == (2, 3, 1, 0)


def test_order_with_init():
    dims = Dims(ndim=3, order=(0, 2, 1))
    assert dims.order == (0, 2, 1)


def test_labels_with_init():
    dims = Dims(ndim=3, axis_labels=('x', 'y', 'z'))
    assert dims.axis_labels == ('x', 'y', 'z')


def test_bad_order():
    dims = Dims(ndim=3)
    with pytest.raises(ValueError):
        dims.order = (0, 0, 1)


def test_pad_bad_labels():
    dims = Dims(ndim=3)
    dims.axis_labels = ('a', 'b')
    assert dims.axis_labels == ('0', 'a', 'b')


def test_keyword_only_dims():
    with pytest.raises(TypeError):
        Dims(3, (1, 2, 3))


def test_point():
    """
    Test point setting.
    """
    dims = Dims(ndim=4)
    assert dims.point == (0,) * 4

    dims.set_range(range(dims.ndim), ((0, 5, 1),) * dims.ndim)
    dims.set_point(3, 4)
    assert dims.point == (0, 0, 0, 4)

    dims.set_point(2, 1)
    assert dims.point == (0, 0, 1, 4)

    dims.set_point((0, 1, 2), (2.1, 2.6, 0.0))
    assert dims.point == (2, 3, 0, 4)


def test_point_variable_step_size():
    dims = Dims(ndim=3)
    assert dims.point == (0,) * 3

    desired_range = ((0, 6, 0.5), (0, 6, 1), (0, 6, 2))
    dims.set_range(range(3), desired_range)
    assert dims.range == desired_range

    # set point updates current_step indirectly
    dims.set_point([0, 1, 2], (2.9, 2.9, 2.9))
    assert dims.current_step == (6, 3, 1)
    # point is a property computed on demand from current_step
    assert dims.point == (3, 3, 2)

    # can set step directly as well
    # note that out of range values get clipped
    dims.set_current_step((0, 1, 2), (1, -3, 5))
    assert dims.current_step == (1, 0, 2)
    assert dims.point == (0.5, 0, 4)

    dims.set_current_step(0, -1)
    assert dims.current_step == (0, 0, 2)
    assert dims.point == (0, 0, 4)

    # mismatched len(axis) vs. len(value)
    with pytest.raises(ValueError):
        dims.set_point((0, 1), (0, 0, 0))

    with pytest.raises(ValueError):
        dims.set_current_step((0, 1), (0, 0, 0))


def test_range():
    """
    Tests range setting.
    """
    dims = Dims(ndim=4)
    assert dims.range == ((0, 2, 1),) * 4

    dims.set_range(3, (0, 4, 2))
    assert dims.range == ((0, 2, 1),) * 3 + ((0, 4, 2),)


def test_range_set_multiple():
    """
    Tests bulk range setting.
    """
    dims = Dims(ndim=4)
    assert dims.range == ((0, 2, 1),) * 4

    dims.set_range((0, 3), [(0, 6, 3), (0, 9, 3)])
    assert dims.range == ((0, 6, 3),) + ((0, 2, 1),) * 2 + ((0, 9, 3),)

    # last_used will be set to the smallest axis in range
    dims.set_range(range(1, 4), ((0, 5, 1),) * 3)
    assert dims.range == ((0, 6, 3),) + ((0, 5, 1),) * 3

    # test with descending axis order
    dims.set_range(axis=(3, 0), _range=[(0, 4, 1), (0, 6, 1)])
    assert dims.range == ((0, 6, 1),) + ((0, 5, 1),) * 2 + ((0, 4, 1),)

    # out of range axis raises a ValueError
    with pytest.raises(ValueError):
        dims.set_range((dims.ndim, 0), [(0.0, 4.0, 1.0)] * 2)

    # sequence lengths for axis and _range do not match
    with pytest.raises(ValueError):
        dims.set_range((0, 1), [(0.0, 4.0, 1.0)] * 3)


def test_axis_labels():
    dims = Dims(ndim=4)
    assert dims.axis_labels == ('0', '1', '2', '3')

    dims.set_axis_label(0, 't')
    assert dims.axis_labels == ('t', '1', '2', '3')

    dims.set_axis_label((0, 1, 3), ('t', 'c', 'last'))
    assert dims.axis_labels == ('t', 'c', '2', 'last')

    # mismatched len(axis) vs. len(value)
    with pytest.raises(ValueError):
        dims.set_point((0, 1), ('x', 'y', 'z'))


def test_order_when_changing_ndim():
    """
    Test order of the dims when changing the number of dimensions.
    """
    dims = Dims(ndim=4)
    dims.set_range(0, (0, 4, 1))
    dims.set_point(0, 2)

    dims.ndim = 5
    # Test that new dims get appended to the beginning of lists
    assert dims.point == (0, 2, 0, 0, 0)
    assert dims.order == (0, 1, 2, 3, 4)
    assert dims.axis_labels == ('0', '1', '2', '3', '4')

    dims.set_range(2, (0, 4, 1))
    dims.set_point(2, 3)
    dims.ndim = 3
    # Test that dims get removed from the beginning of lists
    assert dims.point == (3, 0, 0)
    assert dims.order == (0, 1, 2)
    assert dims.axis_labels == ('2', '3', '4')


def test_labels_order_when_changing_dims():
    dims = Dims(ndim=4)
    dims.ndim = 5
    assert dims.axis_labels == ('0', '1', '2', '3', '4')


@pytest.mark.parametrize(
    "ndim, ax_input, expected", ((2, 1, 1), (2, -1, 1), (4, -3, 1))
)
def test_assert_axis_in_bounds(ndim, ax_input, expected):
    actual = assert_axis_in_bounds(ax_input, ndim)
    assert actual == expected


@pytest.mark.parametrize("ndim, ax_input", ((2, 2), (2, -3)))
def test_assert_axis_out_of_bounds(ndim, ax_input):
    with pytest.raises(ValueError):
        assert_axis_in_bounds(ax_input, ndim)


def test_axis_labels_str_to_list():
    dims = Dims()
    dims.axis_labels = 'TX'
    assert dims.axis_labels == ('T', 'X')


def test_roll():
    """Test basic roll behavior."""
    dims = Dims(ndim=4)
    dims.set_range(0, (0, 10, 1))
    dims.set_range(1, (0, 10, 1))
    dims.set_range(2, (0, 10, 1))
    dims.set_range(3, (0, 10, 1))
    assert dims.order == (0, 1, 2, 3)
    dims._roll()
    assert dims.order == (3, 0, 1, 2)
    dims._roll()
    assert dims.order == (2, 3, 0, 1)


def test_roll_skip_dummy_axis_1():
    """Test basic roll skips axis with length 1."""
    dims = Dims(ndim=4)
    dims.set_range(0, (0, 0, 1))
    dims.set_range(1, (0, 10, 1))
    dims.set_range(2, (0, 10, 1))
    dims.set_range(3, (0, 10, 1))
    assert dims.order == (0, 1, 2, 3)
    dims._roll()
    assert dims.order == (0, 3, 1, 2)
    dims._roll()
    assert dims.order == (0, 2, 3, 1)


def test_roll_skip_dummy_axis_2():
    """Test basic roll skips axis with length 1 when not first."""
    dims = Dims(ndim=4)
    dims.set_range(0, (0, 10, 1))
    dims.set_range(1, (0, 0, 1))
    dims.set_range(2, (0, 10, 1))
    dims.set_range(3, (0, 10, 1))
    assert dims.order == (0, 1, 2, 3)
    dims._roll()
    assert dims.order == (3, 1, 0, 2)
    dims._roll()
    assert dims.order == (2, 1, 3, 0)


def test_roll_skip_dummy_axis_3():
    """Test basic roll skips all axes with length 1."""
    dims = Dims(ndim=4)
    dims.set_range(0, (0, 10, 1))
    dims.set_range(1, (0, 0, 1))
    dims.set_range(2, (0, 10, 1))
    dims.set_range(3, (0, 0, 1))
    assert dims.order == (0, 1, 2, 3)
    dims._roll()
    assert dims.order == (2, 1, 0, 3)
    dims._roll()
    assert dims.order == (0, 1, 2, 3)


def test_changing_focus():
    """Test changing focus updates the last_used prop."""
    # too-few dims, should have no sliders to update
    dims = Dims(ndim=2)
    assert dims.last_used == 0
    dims._focus_down()
    dims._focus_up()
    assert dims.last_used == 0

    dims.ndim = 5
    # Note that with no view attached last used remains
    # None even though new non-displayed dimensions added
    assert dims.last_used == 0
    dims._focus_down()
    assert dims.last_used == 2
    dims._focus_down()
    assert dims.last_used == 1
    dims._focus_up()
    assert dims.last_used == 2
    dims._focus_up()
    assert dims.last_used == 0
    dims._focus_down()
    assert dims.last_used == 2
