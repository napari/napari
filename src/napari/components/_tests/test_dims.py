import pytest
from pydantic import ValidationError

from napari.components import Dims
from napari.components.dims import (
    ensure_axis_in_bounds,
    reorder_after_dim_reduction,
)


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
    assert dims.displayed == (2, 3)
    assert dims.displayed_order == (0, 1)
    assert dims.not_displayed == (0, 1)

    dims.order = (2, 3, 1, 0)
    assert dims.order == (2, 3, 1, 0)
    assert dims.displayed == (1, 0)
    assert dims.displayed_order == (1, 0)
    assert dims.not_displayed == (2, 3)


def test_order_with_init():
    dims = Dims(ndim=3, order=(0, 2, 1))
    assert dims.order == (0, 2, 1)


def test_labels_with_init():
    dims = Dims(ndim=3, axis_labels=('x', 'y', 'z'))
    assert dims.axis_labels == ('x', 'y', 'z')


def test_bad_order():
    dims = Dims(ndim=3)
    with pytest.raises(ValidationError, match='Invalid ordering'):
        dims.order = (0, 0, 1)


def test_pad_bad_labels():
    dims = Dims(ndim=3)
    dims.axis_labels = ('a', 'b')
    assert dims.axis_labels == ('-3', 'a', 'b')


def test_keyword_only_dims():
    with pytest.raises(TypeError):
        Dims(3, (1, 2, 3))


def test_sanitize_input_setters():
    dims = Dims()

    # axis out of range
    with pytest.raises(ValueError, match='not defined for dimensionality'):
        dims._sanitize_input(axis=2, value=3)

    # one value
    with pytest.raises(ValueError, match='cannot set multiple values'):
        dims._sanitize_input(axis=0, value=(1, 2, 3))
    ax, val = dims._sanitize_input(
        axis=0, value=(1, 2, 3), value_is_sequence=True
    )
    assert ax == [0]
    assert val == [(1, 2, 3)]

    # multiple axes
    ax, val = dims._sanitize_input(axis=(0, 1), value=(1, 2))
    assert ax == [0, 1]
    assert val == [1, 2]
    ax, val = dims._sanitize_input(axis=(0, 1), value=((1, 2), (3, 4)))
    assert ax == [0, 1]
    assert val == [(1, 2), (3, 4)]


def test_point():
    """
    Test point setting.
    """
    dims = Dims(ndim=4)
    assert dims.point == (0,) * 4

    dims.range = ((0, 5, 1),) * dims.ndim
    dims.set_point(3, 4)
    assert dims.point == (0, 0, 0, 4)

    dims.set_point(2, 1)
    assert dims.point == (0, 0, 1, 4)

    dims.set_point((0, 1, 2), (2.1, 2.6, 0.0))
    assert dims.point == (2.1, 2.6, 0.0, 4.0)


def test_point_variable_step_size():
    dims = Dims(ndim=3)
    assert dims.point == (0,) * 3

    desired_range = ((0, 6, 0.5), (0, 6, 1), (0, 6, 2))
    dims.range = desired_range
    assert dims.range == desired_range

    # set point updates current_step indirectly
    dims.point = (2.9, 2.9, 2.9)
    assert dims.current_step == (6, 3, 1)
    assert dims.point == (2.9, 2.9, 2.9)

    # can set step directly as well
    # note that out of range values get clipped
    dims.set_current_step((0, 1, 2), (1, -3, 5))
    assert dims.current_step == (1, 0, 3)
    assert dims.point == (0.5, 0, 6)

    dims.set_current_step(0, -1)
    assert dims.current_step == (0, 0, 3)
    assert dims.point == (0, 0, 6)

    # mismatched len(axis) vs. len(value)
    with pytest.raises(ValueError, match='must have equal length'):
        dims.set_point((0, 1), (0, 0, 0))

    with pytest.raises(ValueError, match='must have equal length'):
        dims.set_current_step((0, 1), (0, 0, 0))


def test_range():
    """
    Tests range setting.
    """
    dims = Dims(ndim=4)
    assert dims.range == ((0, 2, 1),) * 4

    dims.set_range(3, (0, 4, 2))
    assert dims.range == ((0, 2, 1),) * 3 + ((0, 4, 2),)

    # start must be lower than stop
    with pytest.raises(ValidationError, match='must be strictly increasing'):
        dims.set_range(0, (1, 0, 1))

    # step must be positive
    with pytest.raises(ValidationError, match='must be strictly positive'):
        dims.set_range(0, (0, 2, 0))
    with pytest.raises(ValidationError, match='must be strictly positive'):
        dims.set_range(0, (0, 2, -1))


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

    # out of range axis raises a ValidationError
    with pytest.raises(ValueError, match='not defined for dimensionality'):
        dims.set_range((dims.ndim, 0), [(0.0, 4.0, 1.0)] * 2)

    # sequence lengths for axis and _range do not match
    with pytest.raises(ValueError, match='must have equal length'):
        dims.set_range((0, 1), [(0.0, 4.0, 1.0)] * 3)


def test_axis_labels():
    dims = Dims(ndim=4)
    assert dims.axis_labels == ('-4', '-3', '-2', '-1')

    dims.set_axis_label(0, 't')
    assert dims.axis_labels == ('t', '-3', '-2', '-1')

    dims.set_axis_label((0, 1, 3), ('t', 'c', 'last'))
    assert dims.axis_labels == ('t', 'c', '-2', 'last')

    # mismatched len(axis) vs. len(value)
    with pytest.raises(ValueError, match='must have equal length'):
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
    assert dims.axis_labels == ('-5', '-4', '-3', '-2', '-1')

    dims.set_range(2, (0, 4, 1))
    dims.set_point(2, 3)
    dims.ndim = 3
    # Test that dims get removed from the beginning of lists
    assert dims.point == (3, 0, 0)
    assert dims.order == (0, 1, 2)
    assert dims.axis_labels == ('-3', '-2', '-1')


def test_labels_order_when_changing_dims():
    dims = Dims(ndim=4)
    dims.ndim = 5
    assert dims.axis_labels == ('-5', '-4', '-3', '-2', '-1')


@pytest.mark.parametrize(
    ('ndim', 'ax_input', 'expected'), [(2, 1, 1), (2, -1, 1), (4, -3, 1)]
)
def test_assert_axis_in_bounds(ndim, ax_input, expected):
    actual = ensure_axis_in_bounds(ax_input, ndim)
    assert actual == expected


@pytest.mark.parametrize(('ndim', 'ax_input'), [(2, 2), (2, -3)])
def test_assert_axis_out_of_bounds(ndim, ax_input):
    with pytest.raises(ValueError, match='not defined for dimensionality'):
        ensure_axis_in_bounds(ax_input, ndim)


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
    dims.roll()
    assert dims.order == (3, 0, 1, 2)
    dims.roll()
    assert dims.order == (2, 3, 0, 1)


def test_roll_skip_dummy_axis_1():
    """Test basic roll skips axis with length 1."""
    dims = Dims(ndim=4)
    dims.set_range(0, (0, 0, 1))
    dims.set_range(1, (0, 10, 1))
    dims.set_range(2, (0, 10, 1))
    dims.set_range(3, (0, 10, 1))
    assert dims.order == (0, 1, 2, 3)
    dims.roll()
    assert dims.order == (0, 3, 1, 2)
    dims.roll()
    assert dims.order == (0, 2, 3, 1)


def test_roll_skip_dummy_axis_2():
    """Test basic roll skips axis with length 1 when not first."""
    dims = Dims(ndim=4)
    dims.set_range(0, (0, 10, 1))
    dims.set_range(1, (0, 0, 1))
    dims.set_range(2, (0, 10, 1))
    dims.set_range(3, (0, 10, 1))
    assert dims.order == (0, 1, 2, 3)
    dims.roll()
    assert dims.order == (3, 1, 0, 2)
    dims.roll()
    assert dims.order == (2, 1, 3, 0)


def test_roll_skip_dummy_axis_3():
    """Test basic roll skips all axes with length 1."""
    dims = Dims(ndim=4)
    dims.set_range(0, (0, 10, 1))
    dims.set_range(1, (0, 0, 1))
    dims.set_range(2, (0, 10, 1))
    dims.set_range(3, (0, 0, 1))
    assert dims.order == (0, 1, 2, 3)
    dims.roll()
    assert dims.order == (2, 1, 0, 3)
    dims.roll()
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


def test_changing_focus_changing_ndisplay():
    dims = Dims(ndim=4, ndisplay=2)
    # simulates putting focus from slider 0 to slider 1
    dims.last_used = 1
    assert dims.last_used == 1
    dims.ndisplay = 3
    # last_used should change from 1 to 0 since dim 1 is displayed now
    assert dims.last_used == 0


def test_floating_point_edge_case():
    # see #4889
    dims = Dims(ndim=2)
    dims.set_range(0, (0.0, 17.665, 3.533))
    assert dims.nsteps[0] == 6


@pytest.mark.parametrize(
    ('order', 'expected'),
    [
        ((0, 1), (0, 1)),  # 2D, increasing, default range
        ((3, 7), (0, 1)),  # 2D, increasing, non-default range
        ((1, 0), (1, 0)),  # 2D, decreasing, default range
        ((5, 2), (1, 0)),  # 2D, decreasing, non-default range
        ((0, 1, 2), (0, 1, 2)),  # 3D, increasing, default range
        ((3, 4, 6), (0, 1, 2)),  # 3D, increasing, non-default range
        ((2, 1, 0), (2, 1, 0)),  # 3D, decreasing, default range
        ((4, 2, 0), (2, 1, 0)),  # 3D, decreasing, non-default range
        ((2, 0, 1), (2, 0, 1)),  # 3D, non-monotonic, default range
        ((4, 0, 1), (2, 0, 1)),  # 3D, non-monotonic, non-default range
    ],
)
def test_reorder_after_dim_reduction(order, expected):
    actual = reorder_after_dim_reduction(order)
    assert actual == expected


def test_nsteps():
    dims = Dims(range=((0, 5, 1), (0, 10, 0.5)))
    assert dims.nsteps == (6, 21)
    dims.nsteps = (11, 11)
    assert dims.range == ((0, 5, 0.5), (0, 10, 1))


def test_thickness():
    dims = Dims(margin_left=(0, 0.5), margin_right=(1, 1))
    assert dims.thickness == (1, 1.5)


def _nav_dims():
    dims = Dims(ndim=4, ndisplay=2)
    dims.range = ((0, 2, 1), (0, 4, 1), (0, 31, 1), (0, 31, 1))
    dims.current_step = (0, 2, 0, 0)
    return dims


def test_navigation_lock_blocks_slice_nav_without_events():
    dims = _nav_dims()
    owner = object()
    assert dims.navigation_locked is False

    events = []
    dims.events.point.connect(lambda e=None: events.append(1))

    dims.lock_navigation(owner)
    assert dims.navigation_locked is True

    dims.set_current_step(1, 4)  # z
    dims.set_point(0, 1.0)  # series, world coords
    assert dims.current_step == (0, 2, 0, 0)  # unchanged
    assert events == []  # a blocked write emits nothing


def test_navigation_lock_exempt_axis_moves():
    dims = _nav_dims()
    owner = object()
    dims.lock_navigation(owner, exempt=(0,))  # series exempt
    dims.set_current_step(0, 1)  # series -> allowed
    dims.set_current_step(1, 4)  # z -> blocked
    assert dims.current_step == (1, 2, 0, 0)


def test_navigation_lock_force_bypasses():
    dims = _nav_dims()
    dims.lock_navigation(object())
    dims.set_current_step(1, 4, force=True)
    assert dims.current_step == (0, 4, 0, 0)


def test_navigation_lock_order_methods():
    dims = Dims(ndim=3)
    dims.lock_navigation(object())  # lock_order defaults True
    before = dims.order
    dims.transpose()
    dims.roll()
    assert dims.order == before  # no-op while locked

    dims2 = Dims(ndim=3)
    dims2.lock_navigation(object(), lock_order=False)
    dims2.transpose()
    assert dims2.order != (0, 1, 2)  # order change allowed when not locked


def test_navigation_lock_ownership():
    dims = _nav_dims()
    a, b = object(), object()
    dims.lock_navigation(a)
    with pytest.raises(RuntimeError):
        dims.lock_navigation(b)  # different owner cannot take it
    with pytest.raises(RuntimeError):
        dims.unlock_navigation(b)  # different owner cannot release it
    dims.lock_navigation(a, exempt=(0,))  # same owner may re-lock
    dims.unlock_navigation(a)
    assert dims.navigation_locked is False
    dims.unlock_navigation(a)  # idempotent no-op when already unlocked


def test_navigation_lock_context_manager():
    dims = _nav_dims()
    owner = object()
    with dims.navigation_lock(owner):
        assert dims.navigation_locked is True
        dims.set_current_step(1, 4)
        assert dims.current_step == (0, 2, 0, 0)
    assert dims.navigation_locked is False
    dims.set_current_step(1, 4)
    assert dims.current_step == (0, 4, 0, 0)


def test_navigation_lock_capability_marker():
    assert getattr(Dims, 'NAVIGATION_LOCK_VERSION', 0) >= 1


def test_navigation_lock_rejects_none_owner():
    dims = _nav_dims()
    with pytest.raises(ValueError, match='owner must not be None'):
        dims.lock_navigation(None)
    assert dims.navigation_locked is False


def test_navigation_lock_force_on_order_methods():
    dims = Dims(ndim=3)
    dims.lock_navigation(object())  # lock_order=True
    before = dims.order
    dims.transpose(force=True)
    assert dims.order != before  # force bypasses the order lock
    rolled = dims.order
    dims.roll(force=True)
    assert dims.order != rolled


def test_navigation_lock_does_not_block_validator_normalization():
    # _check_dims reassigns point/order via direct field assignment under
    # _validating_ctx; the lock guards *methods*, so normalization must still run.
    dims = _nav_dims()
    events = []
    dims.events.point.connect(lambda e=None: events.append(1))
    dims.lock_navigation(object())
    dims.ndim = 5  # triggers _check_dims: pads/normalizes point and order
    assert len(dims.point) == 5
    assert set(dims.order) == set(range(5))
    assert dims.navigation_locked is True


def test_navigation_lock_does_not_guard_direct_assignment():
    # Documented v1 exclusion: direct field/property assignment bypasses the lock
    # (it is also how the validator normalizes). This test pins that contract.
    dims = _nav_dims()
    dims.lock_navigation(object())
    dims.current_step = (1, 3, 0, 0)  # property assignment -> not guarded
    assert dims.current_step == (1, 3, 0, 0)
    dims.order = (1, 0, 2, 3)  # field assignment -> not guarded
    assert tuple(dims.order) == (1, 0, 2, 3)
    dims.ndisplay = 3  # field assignment -> not guarded
    assert dims.ndisplay == 3


def test_navigation_lock_emits_event():
    dims = _nav_dims()
    fired = []
    dims.events.navigation_lock.connect(lambda e=None: fired.append(1))
    owner = object()
    dims.lock_navigation(owner, exempt=(0,))
    assert dims.navigation_lock_exempt == (0,)
    dims.unlock_navigation(owner)
    assert dims.navigation_lock_exempt == ()
    assert len(fired) == 2  # one on lock, one on unlock


def test_navigation_lock_exempt_negative_axis_normalized():
    # A negative exempt axis must be normalized to the same non-negative index
    # set_point compares against; otherwise the intended axis is silently blocked.
    dims = _nav_dims()  # ndim=4
    owner = object()
    dims.lock_navigation(owner, exempt=(-1,))  # last axis
    assert dims.navigation_lock_exempt == (3,)
    dims.set_current_step(3, 5)  # exempt via negative -> allowed
    assert dims.current_step[3] == 5
    dims.set_current_step(-1, 7)  # same axis, negative form -> allowed
    assert dims.current_step[3] == 7
    dims.set_current_step(1, 3)  # non-exempt -> blocked
    assert dims.current_step[1] == 2


def test_navigation_lock_exempt_out_of_range_raises():
    dims = _nav_dims()  # ndim=4
    with pytest.raises(ValueError, match='not defined for dimensionality'):
        dims.lock_navigation(object(), exempt=(99,))
    assert dims.navigation_locked is False  # rejected, no partial lock


def test_navigation_lock_owner_property():
    dims = _nav_dims()
    owner = object()
    assert dims.navigation_lock_owner is None
    dims.lock_navigation(owner)
    assert dims.navigation_lock_owner is owner
    dims.unlock_navigation(owner)
    assert dims.navigation_lock_owner is None


def test_navigation_lock_unlock_returns_state_to_defaults():
    # Invariant: releasing the lock returns the config to its unlocked defaults
    # (symmetry between _nav_lock_exempt and _nav_lock_order). This is cosmetic
    # cleanup, not a functional guard -- a fresh lock overwrites _nav_lock_order
    # regardless -- so this only pins the unlocked-state defaults.
    dims = Dims(ndim=3)
    owner = object()
    dims.lock_navigation(owner, exempt=(0,), lock_order=False)
    dims.unlock_navigation(owner)
    assert dims.navigation_lock_exempt == ()
    assert dims._nav_lock_order is True


def test_navigation_lock_nested_context_manager():
    # A nested same-owner context must restore the outer lock on inner exit,
    # not release it wholesale.
    dims = _nav_dims()
    owner = object()
    with dims.navigation_lock(owner, exempt=(0,)):
        assert dims.navigation_lock_exempt == (0,)
        with dims.navigation_lock(owner, exempt=(1,)):
            assert dims.navigation_lock_exempt == (1,)
        # inner exit restores the outer config, does NOT unlock
        assert dims.navigation_locked is True
        assert dims.navigation_lock_exempt == (0,)
    # outermost exit releases
    assert dims.navigation_locked is False


# --- per-axis (persistent, user-facing) navigation locks ---------------------


def test_lock_axis_blocks_nav_without_events():
    dims = _nav_dims()
    events = []
    dims.events.point.connect(lambda e=None: events.append(1))

    dims.lock_axis(1)  # z
    assert dims.axis_locked == (False, True, False, False)

    dims.set_current_step(1, 4)
    dims.set_point(1, 3.0)
    assert dims.current_step == (0, 2, 0, 0)  # unchanged
    assert events == []  # a blocked write emits nothing


def test_lock_axis_leaves_other_axes_movable():
    dims = _nav_dims()
    dims.lock_axis(1)
    dims.set_current_step(0, 1)  # unlocked -> moves
    dims.set_current_step(1, 4)  # locked -> blocked
    assert dims.current_step == (1, 2, 0, 0)


def test_lock_axis_by_name():
    dims = _nav_dims()
    dims.axis_labels = ('time', 'z', 'y', 'x')
    dims.lock_axis('time')
    assert dims.axis_locked == (True, False, False, False)
    dims.set_current_step(0, 1)
    assert dims.current_step[0] == 0  # blocked
    dims.unlock_axis('time')
    dims.set_current_step(0, 1)
    assert dims.current_step[0] == 1


def test_lock_axis_unknown_name_raises():
    dims = _nav_dims()
    dims.axis_labels = ('time', 'z', 'y', 'x')
    with pytest.raises(ValueError, match='No axis named'):
        dims.lock_axis('channel')


def test_lock_axis_ambiguous_name_raises():
    # axis_labels are not unique, so a duplicated name has no single answer.
    dims = Dims(ndim=2, axis_labels=('y', 'y'))
    with pytest.raises(ValueError, match='ambiguous'):
        dims.lock_axis('y')


def test_lock_axis_negative_index_normalized():
    dims = _nav_dims()  # ndim=4
    dims.lock_axis(-1)
    assert dims.axis_locked == (False, False, False, True)
    dims.set_current_step(3, 5)
    assert dims.current_step[3] == 0  # blocked


def test_lock_axis_out_of_range_raises():
    dims = _nav_dims()  # ndim=4
    with pytest.raises(ValueError, match='not defined for dimensionality'):
        dims.lock_axis(99)
    assert dims.axis_locked == (False,) * 4  # rejected, no partial state


def test_lock_axis_force_bypasses():
    dims = _nav_dims()
    dims.lock_axis(1)
    dims.set_current_step(1, 4, force=True)
    assert dims.current_step == (0, 4, 0, 0)


def test_lock_all_and_unlock_all_axes():
    dims = _nav_dims()
    dims.lock_all_axes()
    assert dims.axis_locked == (True,) * 4
    dims.set_current_step(1, 4)
    assert dims.current_step == (0, 2, 0, 0)  # every axis blocked
    dims.unlock_all_axes()
    assert dims.axis_locked == (False,) * 4
    dims.set_current_step(1, 4)
    assert dims.current_step == (0, 4, 0, 0)


def test_axis_locked_emits_event():
    dims = _nav_dims()
    fired = []
    dims.events.axis_locked.connect(lambda e=None: fired.append(1))
    dims.lock_axis(0)
    dims.unlock_axis(0)
    assert len(fired) == 2


def test_owner_lock_governs_and_suspends_per_axis_locks():
    # While an owner lock is held its exempt set alone decides movability; the
    # sticky per-axis pin is suspended for the duration and resumes on release.
    dims = _nav_dims()
    dims.lock_axis(0)  # user pins axis 0
    owner = object()

    dims.lock_navigation(owner, exempt=(0,))
    dims.set_current_step(0, 1)  # exempt -> moves despite the user pin
    assert dims.current_step[0] == 1
    dims.set_current_step(1, 4)  # not exempt -> blocked
    assert dims.current_step[1] == 2

    dims.unlock_navigation(owner)
    dims.set_current_step(0, 2)  # user pin back in force -> blocked
    assert dims.current_step[0] == 1


def test_per_axis_lock_mutation_during_owner_lock_raises():
    # The per-axis configuration is frozen while an operation owns navigation.
    dims = _nav_dims()
    owner = object()
    dims.lock_navigation(owner)
    for call in (
        lambda: dims.lock_axis(0),
        lambda: dims.unlock_axis(0),
        dims.lock_all_axes,
        dims.unlock_all_axes,
    ):
        with pytest.raises(RuntimeError, match='while navigation is locked'):
            call()
    assert dims.axis_locked == (False,) * 4  # unchanged


def test_axis_locked_left_pads_on_ndim_change():
    # New axes are prepended, so a lock must track its axis, not its index.
    dims = _nav_dims()  # ndim=4
    dims.lock_axis(3)
    assert dims.axis_locked == (False, False, False, True)
    dims.ndim = 5
    assert dims.axis_locked == (False, False, False, False, True)
    dims.ndim = 4
    assert dims.axis_locked == (False, False, False, True)


def test_per_axis_lock_does_not_guard_direct_assignment():
    # Documented v1 escape hatch: the per-axis lock inherits the owner lock's
    # method-only contract, so direct field/property assignment bypasses it.
    # This test pins that contract.
    dims = _nav_dims()
    dims.lock_axis(1)
    dims.current_step = (1, 3, 0, 0)  # property assignment -> not guarded
    assert dims.current_step == (1, 3, 0, 0)
    dims.point = (0.0, 1.0, 0.0, 0.0)  # field assignment -> not guarded
    assert dims.point == (0.0, 1.0, 0.0, 0.0)


def test_per_axis_lock_does_not_block_order_changes():
    # Scope: the per-axis lock governs slice position only. Axis-order changes
    # stay governed by the owner lock's lock_order.
    dims = _nav_dims()
    dims.lock_all_axes()
    dims.roll()
    assert tuple(dims.order) == (3, 0, 1, 2)
    dims.order = (0, 1, 2, 3)
    dims.transpose()
    assert tuple(dims.order) == (0, 1, 3, 2)


def test_axis_lock_interactive_does_not_gate_programmatic_locking():
    # The switch gates only the UI click path; methods are always allowed.
    dims = _nav_dims()
    dims.axis_lock_interactive = False
    dims.lock_axis(1)
    assert dims.axis_locked == (False, True, False, False)
    dims.set_current_step(1, 4)
    assert dims.current_step[1] == 2  # still enforced


def test_focus_skips_locked_axes():
    dims = _nav_dims()  # not_displayed == (0, 1), both have nsteps > 1
    dims.lock_axis(0)
    dims.last_used = 0  # focused axis is the locked one
    dims._focus_up()
    assert dims.last_used == 1  # skipped the locked axis
    dims._focus_up()
    assert dims.last_used == 1  # only one candidate left
    dims._focus_down()
    assert dims.last_used == 1


def test_focus_noop_when_all_axes_locked():
    dims = _nav_dims()
    dims.lock_all_axes()
    dims.last_used = 1
    dims._focus_up()
    assert dims.last_used == 1
    dims._focus_down()
    assert dims.last_used == 1


def test_locking_inactive_axis_leaves_active_axis_alone():
    # Locking a slider you are not on must not steal focus.
    dims = _nav_dims()  # not_displayed == (0, 1)
    dims.last_used = 0
    dims.lock_axis(1)
    assert dims.last_used == 0


def test_locking_active_axis_moves_active_to_a_movable_one():
    # The active slider should always be one you can actually move.
    dims = _nav_dims()
    dims.last_used = 0
    dims.lock_axis(0)
    assert dims.last_used == 1


def test_unlocking_makes_axis_active_when_none_are_movable():
    # With every slider locked, last_used still names a real slider; unlocking
    # one makes it the only candidate, so it becomes active.
    dims = _nav_dims()
    dims.lock_all_axes()
    assert dims.last_used in (0, 1)  # still a real slider, not None
    dims.unlock_axis(1)
    assert dims.last_used == 1


def test_unlocking_does_not_steal_active_axis_when_one_is_movable():
    dims = _nav_dims()
    dims.lock_axis(1)
    dims.last_used = 0
    dims.unlock_axis(1)
    assert dims.last_used == 0  # axis 0 was already movable and active
