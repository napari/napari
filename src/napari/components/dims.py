import contextlib
from collections.abc import Sequence
from numbers import Integral
from typing import (
    Any,
    ClassVar,
    Literal,
    NamedTuple,
)

import numpy as np
import pint
from pydantic import field_validator, model_validator

from napari.utils.events import EventedModel
from napari.utils.events.event import Event
from napari.utils.misc import argsort, reorder_after_dim_reduction
from napari.utils.translations import trans


class RangeTuple(NamedTuple):
    start: float
    stop: float
    step: float


class Dims(EventedModel):
    """Dimensions object modeling slicing and displaying.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    ndisplay : int
        Number of displayed dimensions.
    range : tuple of 3-tuple of float
        List of tuples (min, max, step), one for each dimension in world
        coordinates space. Lower and upper bounds are inclusive.
    point : tuple of floats
        Dims position in world coordinates for each dimension.
    margin_left : tuple of floats
        Left margin in world pixels of the slice for each dimension.
    margin_right : tuple of floats
        Right margin in world pixels of the slice for each dimension.
    order : tuple of int
        Tuple of ordering the dimensions, where the last dimensions are rendered.
    axis_labels : tuple of str
        Tuple of labels for each dimension.
    units : tuple of pint.Unit, optional
        Shared world units for each dimension.
        If ``None``, no additional unit conversion is applied.
    last_used : int
        Dimension which was last interacted with.
    axis_locked : tuple of bool
        Per-axis persistent navigation lock. If True, navigation cannot move
        that axis's slice position. See ``lock_axis``.
    axis_lock_interactive : bool
        Whether the user may toggle per-axis locks through the UI. Programmatic
        ``lock_axis``/``unlock_axis`` are unaffected by this. Default True.

    Attributes
    ----------
    ndim : int
        Number of dimensions.
    ndisplay : int
        Number of displayed dimensions.
    range : tuple of 3-tuple of float
        List of tuples (min, max, step), one for each dimension in world
        coordinates space. Lower and upper bounds are inclusive.
    point : tuple of floats
        Dims position in world coordinates for each dimension.
    margin_left : tuple of floats
        Left margin (=thickness) in world pixels of the slice for each dimension.
    margin_right : tuple of floats
        Right margin (=thickness) in world pixels of the slice for each dimension.
    order : tuple of int
        Tuple of ordering the dimensions, where the last dimensions are rendered.
    axis_labels : tuple of str
        Tuple of labels for each dimension.
    units : tuple of pint.Unit or None
        Shared world units for each dimension.
        If ``None``, no additional unit conversion is applied.
    last_used : int
        Dimension which was last used.
        Tuple the slider position for each dims slider, in world coordinates.
    current_step : tuple of int
        Current step for each dimension (same as point, but in slider coordinates).
    nsteps : tuple of int
        Number of steps available to each slider. These are calculated from
        the ``range``.
    thickness : tuple of floats
        Thickness of the slice (sum of both margins) for each dimension in world coordinates.
    displayed : tuple of int
        List of dimensions that are displayed. These are calculated from the
        ``order`` and ``ndisplay``.
    not_displayed : tuple of int
        List of dimensions that are not displayed. These are calculated from the
        ``order`` and ``ndisplay``.
    displayed_order : tuple of int
        Order of only displayed dimensions. These are calculated from the
        ``displayed`` dimensions.
    rollable :  tuple of bool
        Tuple of axis roll state. If True the axis is rollable.
    axis_locked : tuple of bool
        Tuple of per-axis persistent navigation lock state. If True the axis's
        slice position is locked. See ``lock_axis``.
    axis_lock_interactive : bool
        Whether the user may toggle per-axis locks through the UI.
    """

    # fields
    ndim: int = 2
    ndisplay: Literal[2, 3] = 2

    order: tuple[int, ...] = ()
    axis_labels: tuple[str, ...] = ()
    rollable: tuple[bool, ...] = ()
    axis_locked: tuple[bool, ...] = ()

    range: tuple[RangeTuple, ...] = ()
    margin_left: tuple[float, ...] = ()
    margin_right: tuple[float, ...] = ()
    point: tuple[float, ...] = ()
    units: tuple[pint.Unit, ...] | None = None

    last_used: int = 0
    # Whether the user may toggle per-axis locks through the UI. Gates only the
    # UI click path; programmatic lock_axis/unlock_axis are always allowed.
    axis_lock_interactive: bool = True

    # Capability marker for the navigation lock (see lock_navigation). Downstream
    # code should feature-detect the *contract version*, not method presence:
    #   getattr(type(dims), 'NAVIGATION_LOCK_VERSION', 0) >= 1
    # Version 1 guards navigation through the *methods* only: set_point and
    # everything that funnels through it (set_current_step, _increment_dims_*) and,
    # when lock_order is set, order changes via roll()/transpose(). It does NOT
    # guard direct field/property assignment — `dims.point = ...`,
    # `dims.current_step = ...`, `dims.order = ...`, `dims.ndisplay = ...` — which
    # bypass the lock by design (they are also how the internal validator
    # normalizes state; see _check_dims). Callers who expose those assignment paths
    # are responsible for gating them.
    NAVIGATION_LOCK_VERSION: ClassVar[int] = 1

    # private vars
    _play_ready: bool = True  # False if currently awaiting a draw event
    _scroll_progress: int = 0
    _validating: bool = False
    # Navigation lock: None when unlocked, else the owning object. `_nav_lock_exempt`
    # lists axes still movable while locked; `_nav_lock_order` locks roll/transpose.
    _nav_lock_owner: Any = None
    _nav_lock_exempt: tuple[int, ...] = ()
    _nav_lock_order: bool = True

    # validators
    # check fields is false to allow private fields to work
    @field_validator(
        'order',
        'axis_labels',
        'rollable',
        'axis_locked',
        'point',
        'margin_left',
        'margin_right',
        mode='before',
    )
    def _as_tuple(v):
        return tuple(v)

    @field_validator('range', mode='before')
    def _check_ranges(ranges):
        """
        Ensure the range values are sane.

        - start < stop
        - step > 0
        """
        for axis, (start, stop, step) in enumerate(ranges):
            if start > stop:
                raise ValueError(
                    trans._(
                        'start and stop must be strictly increasing, but got ({start}, {stop}) for axis {axis}',
                        deferred=True,
                        start=start,
                        stop=stop,
                        axis=axis,
                    )
                )
            if step <= 0:
                raise ValueError(
                    trans._(
                        'step must be strictly positive, but got {step} for axis {axis}.',
                        deferred=True,
                        step=step,
                        axis=axis,
                    )
                )
        return ranges

    @model_validator(mode='after')
    def _check_dims(self):
        """Check the consistency of dimensionality for all attributes.

        Parameters
        ----------
        values : dict
            Values dictionary to update dims model with.
        """
        if self._validating:
            return self
        with self.events.blocker_all(), self._validating_ctx():
            ndim = self.ndim

            range_ = ensure_len(self.range, ndim, pad_width=(0.0, 2.0, 1.0))
            self.range = tuple(RangeTuple(*rng) for rng in range_)

            point = ensure_len(self.point, ndim, pad_width=0.0)
            # ensure point is limited to range
            self.point = tuple(
                np.clip(pt, rng.start, rng.stop)
                for pt, rng in zip(point, self.range, strict=False)
            )

            self.margin_left = ensure_len(
                self.margin_left, ndim, pad_width=0.0
            )
            self.margin_right = ensure_len(
                self.margin_right, ndim, pad_width=0.0
            )

            # order and label default computation is too different to include in ensure_len()
            # Check the order tuple has same number of elements as ndim
            order = self.order
            if len(order) < ndim:
                order_ndim = len(order)
                # new dims are always prepended
                prepended_dims = tuple(range(ndim - order_ndim))
                # maintain existing order, but shift accordingly
                existing_order = tuple(o + ndim - order_ndim for o in order)
                order = prepended_dims + existing_order
            elif len(order) > ndim:
                order = reorder_after_dim_reduction(order[-ndim:])
            self.order = order

            # Check the order is a permutation of 0, ..., ndim - 1
            if set(self.order) != set(range(ndim)):
                raise ValueError(
                    trans._(
                        'Invalid ordering {order} for {ndim} dimensions',
                        deferred=True,
                        order=self.order,
                        ndim=ndim,
                    )
                )

        # Check the axis labels tuple has same number of elements as ndim
        axis_labels = self.axis_labels
        labels_ndim = len(axis_labels)
        if labels_ndim < ndim:
            # Append new "default" labels to existing ones

            self.axis_labels = (
                tuple(map(str, range(-ndim, -labels_ndim))) + axis_labels
            )
        elif labels_ndim > ndim:
            self.axis_labels = axis_labels[-ndim:]

        with self._validating_ctx():
            # Check the rollable axes tuple has same number of elements as ndim
            self.rollable = ensure_len(self.rollable, ndim, True)
            # New axes are unlocked by default; left-pad like the other per-axis
            # tuples so a lock tracks its axis across ndim changes.
            self.axis_locked = ensure_len(self.axis_locked, ndim, False)

        # If the last used slider is no longer visible -- or can no longer be
        # moved, since marking a locked slider as active is pointless -- move to
        # another one. Falls back to the visible sliders when every one of them
        # is locked, so last_used always names a real slider; unlocking one then
        # makes it active, because it becomes the only candidate.
        last_used = self.last_used
        ndisplay = self.ndisplay
        dims_range = self.range
        nsteps = self._nsteps_from_range(dims_range)
        not_displayed = [
            d for d in order[:-ndisplay] if len(nsteps) > d and nsteps[d] > 1
        ]
        movable = [d for d in not_displayed if self._axis_movable(d)]
        candidates = movable or not_displayed
        if len(candidates) > 0 and last_used not in candidates:
            self.last_used = candidates[0]

        return self

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Not a field: an event fired when the navigation lock engages/releases,
        # so views (e.g. the Qt dim sliders) can disable themselves while locked.
        self.events.add(navigation_lock=Event)

    @staticmethod
    def _nsteps_from_range(dims_range) -> tuple[float, ...]:
        return tuple(
            # "or 1" ensures degenerate dimension works
            int((rng.stop - rng.start) / (rng.step or 1)) + 1
            for rng in dims_range
        )

    @property
    def nsteps(self) -> tuple[float, ...]:
        return self._nsteps_from_range(self.range)

    @nsteps.setter
    def nsteps(self, value):
        self.range = tuple(
            RangeTuple(
                rng.start, rng.stop, (rng.stop - rng.start) / (nsteps - 1)
            )
            for rng, nsteps in zip(self.range, value, strict=False)
        )

    @property
    def current_step(self):
        return tuple(
            round((point - rng.start) / (rng.step or 1))
            for point, rng in zip(self.point, self.range, strict=False)
        )

    @current_step.setter
    def current_step(self, value):
        self.point = tuple(
            rng.start + point * (rng.step or 1)
            for point, rng in zip(value, self.range, strict=False)
        )

    @property
    def margin_left_step(self) -> tuple[int, ...]:
        return tuple(
            round(mrg / (rng.step or 1))
            for mrg, rng in zip(self.margin_left, self.range, strict=False)
        )

    @margin_left_step.setter
    def margin_left_step(self, value):
        self.margin_left = tuple(
            mrg * (rng.step or 1)
            for mrg, rng in zip(value, self.range, strict=False)
        )

    @property
    def margin_right_step(self) -> tuple[int, ...]:
        return tuple(
            round(mrg / (rng.step or 1))
            for mrg, rng in zip(self.margin_right, self.range, strict=False)
        )

    @margin_right_step.setter
    def margin_right_step(self, value):
        self.margin_right = tuple(
            mrg * (rng.step or 1)
            for mrg, rng in zip(value, self.range, strict=False)
        )

    @property
    def thickness(self) -> tuple[float, ...]:
        return tuple(
            left + right
            for left, right in zip(
                self.margin_left, self.margin_right, strict=False
            )
        )

    @thickness.setter
    def thickness(self, value):
        self.margin_left = self.margin_right = tuple(val / 2 for val in value)

    @property
    def thickness_step(self) -> tuple[int, ...]:
        return tuple(
            left + right
            for left, right in zip(
                self.margin_left_step, self.margin_right_step, strict=False
            )
        )

    @thickness_step.setter
    def thickness_step(self, value):
        self.margin_left_step = self.margin_right_step = tuple(
            val // 2 for val in value
        )

    @property
    def displayed(self) -> tuple[int, ...]:
        """Tuple: Dimensions that are displayed."""
        return self.order[-self.ndisplay :]

    @property
    def not_displayed(self) -> tuple[int, ...]:
        """Tuple: Dimensions that are not displayed."""
        return self.order[: -self.ndisplay]

    @property
    def displayed_order(self) -> tuple[int, ...]:
        return tuple(argsort(self.displayed))

    def set_range(
        self,
        axis: int | Sequence[int],
        _range: Sequence[int | float] | Sequence[Sequence[int | float]],
    ):
        """Sets ranges (min, max, step) for the given dimensions.

        Parameters
        ----------
        axis : int or sequence of int
            Dimension index or a sequence of axes whos range will be set.
        _range : tuple or sequence of tuple
            Range specified as (min, max, step) or a sequence of these range
            tuples.
        """
        axis, value = self._sanitize_input(
            axis, _range, value_is_sequence=True
        )
        full_range = list(self.range)
        for ax, val in zip(axis, value, strict=False):
            full_range[ax] = val
        self.range = tuple(full_range)

    def set_point(
        self,
        axis: int | Sequence[int],
        value: float | Sequence[float],
        *,
        force: bool = False,
    ):
        """Sets point to slice dimension in world coordinates.

        Parameters
        ----------
        axis : int or sequence of int
            Dimension index or a sequence of axes whos point will be set.
        value : scalar or sequence of scalars
            Value of the point for each axis.
        force : bool
            When True, bypass the navigation lock (see ``lock_navigation``).
            Defaults to False, so locked axes are silently skipped.
        """
        axis, value = self._sanitize_input(
            axis, value, value_is_sequence=False
        )
        if not force:
            allowed = [
                (ax, val)
                for ax, val in zip(axis, value, strict=False)
                if self._axis_movable(ax)
            ]
            # Fully blocked: no-op, and crucially emit no event, so a blocked
            # write cannot feed a dims-change listener loop.
            if not allowed:
                return
            axis = [ax for ax, _ in allowed]
            value = [val for _, val in allowed]
        full_point = list(self.point)
        for ax, val in zip(axis, value, strict=False):
            full_point[ax] = val
        self.point = tuple(full_point)

    def set_current_step(
        self,
        axis: int | Sequence[int],
        value: int | Sequence[int],
        *,
        force: bool = False,
    ):
        axis, value = self._sanitize_input(
            axis, value, value_is_sequence=False
        )
        range_ = list(self.range)
        value_world = []
        for ax, val in zip(axis, value, strict=False):
            rng = range_[ax]
            value_world.append(rng.start + val * rng.step)
        self.set_point(axis, value_world, force=force)

    def set_axis_label(
        self,
        axis: int | Sequence[int],
        label: str | Sequence[str],
    ):
        """Sets new axis labels for the given axes.

        Parameters
        ----------
        axis : int or sequence of int
            Dimension index or a sequence of axes whos labels will be set.
        label : str or sequence of str
            Given labels for the specified axes.
        """
        axis, label = self._sanitize_input(
            axis, label, value_is_sequence=False
        )
        full_axis_labels = list(self.axis_labels)
        for ax, val in zip(axis, label, strict=False):
            full_axis_labels[ax] = val
        self.axis_labels = tuple(full_axis_labels)

    def reset(self):
        """Reset dims values to initial states."""
        # Don't reset axis labels
        # TODO: could be optimized with self.update, but need to fix
        #       event firing in EventedModel first
        self.range = ((0, 2, 1),) * self.ndim
        self.point = (0,) * self.ndim
        self.order = tuple(range(self.ndim))
        self.margin_left = (0,) * self.ndim
        self.margin_right = (0,) * self.ndim
        self.rollable = (True,) * self.ndim

    def transpose(self, *, force: bool = False):
        """Transpose displayed dimensions.

        This swaps the order of the last two displayed dimensions.
        The order of the displayed is taken from Dims.order.

        Pass ``force=True`` to bypass an active navigation lock.
        """
        if (
            self._nav_lock_owner is not None
            and self._nav_lock_order
            and not force
        ):
            return
        order = list(self.order)
        order[-2], order[-1] = order[-1], order[-2]
        self.order = order

    def _increment_dims_right(self, axis: int | None = None):
        """Increment dimensions to the right along given axis, or last used axis if None

        Parameters
        ----------
        axis : int, optional
            Axis along which to increment dims, by default None
        """
        if axis is None:
            axis = self.last_used
        self.set_current_step(axis, self.current_step[axis] + 1)

    def _increment_dims_left(self, axis: int | None = None):
        """Increment dimensions to the left along given axis, or last used axis if None

        Parameters
        ----------
        axis : int, optional
            Axis along which to increment dims, by default None
        """
        if axis is None:
            axis = self.last_used
        self.set_current_step(axis, self.current_step[axis] - 1)

    def _focus_up(self):
        """Shift focused dimension slider to be the next slider above."""
        # Skip locked axes: focusing a slider you cannot move is pointless.
        sliders = [
            d
            for d in self.not_displayed
            if self.nsteps[d] > 1 and self._axis_movable(d)
        ]
        if len(sliders) == 0:
            return

        # last_used may not be a candidate (e.g. it is the locked axis), so the
        # index() below would raise; fall back to the first candidate.
        if self.last_used in sliders:
            index = (sliders.index(self.last_used) + 1) % len(sliders)
        else:
            index = 0
        self.last_used = sliders[index]

    def _focus_down(self):
        """Shift focused dimension slider to be the next slider bellow."""
        # Skip locked axes: focusing a slider you cannot move is pointless.
        sliders = [
            d
            for d in self.not_displayed
            if self.nsteps[d] > 1 and self._axis_movable(d)
        ]
        if len(sliders) == 0:
            return

        # last_used may not be a candidate (e.g. it is the locked axis), so the
        # index() below would raise; fall back to the last candidate.
        if self.last_used in sliders:
            index = (sliders.index(self.last_used) - 1) % len(sliders)
        else:
            index = -1
        self.last_used = sliders[index]

    def roll(self, *, force: bool = False):
        """Roll order of dimensions for display.

        Pass ``force=True`` to bypass an active navigation lock.
        """
        if (
            self._nav_lock_owner is not None
            and self._nav_lock_order
            and not force
        ):
            return
        order = np.array(self.order)
        # we combine "rollable" and "nsteps" into a mask for rolling
        # this mask has to be aligned to "order" as "rollable" and
        # "nsteps" are static but order is dynamic, meaning "rollable"
        # and "nsteps" encode the axes by position, whereas "order"
        # encodes axis by number
        valid = np.logical_and(self.rollable, np.array(self.nsteps) > 1)[order]
        order[valid] = np.roll(order[valid], shift=1)
        self.order = order

    def _go_to_center_step(self):
        self.current_step = [int((ns - 1) / 2) for ns in self.nsteps]

    @property
    def navigation_locked(self) -> bool:
        """bool: whether slice navigation is currently locked.

        See ``lock_navigation``. While locked, ``set_point`` /
        ``set_current_step`` (and everything that funnels through them, i.e. the
        sliders, wheel and arrow-key stepping) are no-ops for non-exempt axes,
        and — when locked with ``lock_order`` — ``roll``/``transpose`` are no-ops.
        Pass ``force=True`` to ``set_point``/``set_current_step`` to bypass.
        """
        return self._nav_lock_owner is not None

    @property
    def navigation_lock_exempt(self) -> tuple[int, ...]:
        """Axes that stay navigable while navigation is locked (see lock_navigation)."""
        return self._nav_lock_exempt

    @property
    def navigation_lock_owner(self) -> Any:
        """The object currently holding the navigation lock, or None if unlocked.

        See ``lock_navigation``. Exposed read-only so a caller can check whether
        it (or an object it manages) holds the lock before releasing it, e.g. to
        avoid the ``RuntimeError`` ``unlock_navigation`` raises for a non-owner.
        """
        return self._nav_lock_owner

    def _axis_movable(self, axis: int, *, force: bool = False) -> bool:
        """Whether navigation may move ``axis`` right now.

        Composes the two lock tiers as a precedence ladder. ``force`` bypasses
        everything. While a ``lock_navigation`` owner lock is held, the owner's
        exempt set alone governs and the persistent per-axis locks are
        *suspended* for the duration (the owner is the active operation and its
        request is authoritative). With no owner, the sticky per-axis
        ``axis_locked`` state governs.

        ``axis`` must be a canonical non-negative index (as produced by
        ``_sanitize_input``/``ensure_axis_in_bounds``).
        """
        if force:
            return True
        if self._nav_lock_owner is not None:
            return axis in self._nav_lock_exempt
        return not self.axis_locked[axis]

    def is_axis_movable(self, axis: int | str) -> bool:
        """Whether navigation may currently move ``axis``'s slice position.

        Public form of the composed lock state: True unless the axis is held by
        a persistent per-axis lock (``lock_axis``) or by an active
        ``lock_navigation`` owner lock. Views use this to decide whether an
        axis's navigation controls should be enabled, so the precedence between
        the two lock tiers lives in one place.

        Parameters
        ----------
        axis : int or str
            Axis index or an axis label (see ``lock_axis``).
        """
        return self._axis_movable(self._normalize_axis(axis))

    def lock_navigation(
        self,
        owner: Any,
        *,
        exempt: Sequence[int] = (),
        lock_order: bool = True,
    ) -> None:
        """Lock slice navigation until ``unlock_navigation`` is called.

        Intended for an application that must freeze the viewed slice during an
        operation (e.g. drawing a shape keyed to the current slice). A single
        owner holds the lock at a time.

        Parameters
        ----------
        owner : Any
            The object taking the lock (must not be None). ``unlock_navigation``
            must be called with the same object. Acquiring while a *different*
            owner holds the lock raises ``RuntimeError``.
        exempt : sequence of int
            Axes that remain freely navigable while locked (e.g. a parametric
            axis the application chooses to allow).
        lock_order : bool
            Also lock ``roll``/``transpose`` (axis-order changes). Default True.
        """
        if owner is None:
            # None is the unlocked sentinel; accepting it would silently no-op.
            raise ValueError(
                trans._(
                    'Navigation lock owner must not be None.', deferred=True
                )
            )
        if (
            self._nav_lock_owner is not None
            and self._nav_lock_owner is not owner
        ):
            raise RuntimeError(
                trans._(
                    'Dims navigation is already locked by another owner.',
                    deferred=True,
                )
            )
        # Normalize exempt axes to canonical non-negative indices, validating
        # each is in range *before* mutating any lock state. Without this,
        # exempt=(-1,) would never match the normalized axes set_point compares
        # against (silently blocking the axis the caller meant to free), and an
        # out-of-range axis would be accepted silently. ensure_axis_in_bounds
        # raises ValueError on out-of-range; doing it first keeps a rejected
        # call from leaving a partial lock.
        exempt_normalized = tuple(
            sorted({ensure_axis_in_bounds(ax, self.ndim) for ax in exempt})
        )
        self._nav_lock_owner = owner
        self._nav_lock_exempt = exempt_normalized
        self._nav_lock_order = lock_order
        self.events.navigation_lock()

    def unlock_navigation(self, owner: Any) -> None:
        """Release a navigation lock taken by ``owner``.

        A no-op if navigation is not locked. Raises ``RuntimeError`` if a
        *different* owner holds the lock, so one owner cannot release another's.
        """
        if self._nav_lock_owner is None:
            return
        if self._nav_lock_owner is not owner:
            raise RuntimeError(
                trans._(
                    'Dims navigation is locked by a different owner.',
                    deferred=True,
                )
            )
        self._nav_lock_owner = None
        self._nav_lock_exempt = ()
        # Cosmetic symmetry with _nav_lock_exempt: return the unlocked state to
        # defaults. Not load-bearing — every lock_navigation overwrites
        # _nav_lock_order, and the order guards check the owner first, so a
        # lingering value is never observed while unlocked.
        self._nav_lock_order = True
        self.events.navigation_lock()

    @contextlib.contextmanager
    def navigation_lock(
        self,
        owner: Any,
        *,
        exempt: Sequence[int] = (),
        lock_order: bool = True,
    ):
        """Context manager wrapping ``lock_navigation``/``unlock_navigation``.

        Re-entrant for a single owner: nesting ``with dims.navigation_lock(owner)``
        blocks restore the *outer* lock's configuration on exit rather than
        releasing the lock, so an inner block with different ``exempt``/
        ``lock_order`` does not strand the outer block unlocked. The outermost
        block releases the lock.
        """
        # Snapshot the lock state we are about to overwrite so we can restore it
        # (rather than fully unlocking) when unwinding a nested acquisition.
        prev_owner = self._nav_lock_owner
        prev_exempt = self._nav_lock_exempt
        prev_order = self._nav_lock_order
        self.lock_navigation(owner, exempt=exempt, lock_order=lock_order)
        try:
            yield
        finally:
            if prev_owner is None:
                # We were the outermost acquisition: release the lock.
                self.unlock_navigation(owner)
            else:
                # Nested: hand control back to the enclosing block's config.
                self._nav_lock_owner = prev_owner
                self._nav_lock_exempt = prev_exempt
                self._nav_lock_order = prev_order
                self.events.navigation_lock()

    def _normalize_axis(self, axis: int | str) -> int:
        """Resolve an axis given as an index or an ``axis_labels`` name.

        Raises ``ValueError`` for an out-of-range index, an unknown name, or a
        name that matches more than one axis (labels are not unique).
        """
        if isinstance(axis, str):
            matches = [
                i for i, label in enumerate(self.axis_labels) if label == axis
            ]
            if not matches:
                raise ValueError(
                    trans._(
                        'No axis named {name}.', deferred=True, name=axis
                    )
                )
            if len(matches) > 1:
                raise ValueError(
                    trans._(
                        'Axis name {name} is ambiguous; it matches axes {matches}.',
                        deferred=True,
                        name=axis,
                        matches=matches,
                    )
                )
            return matches[0]
        return ensure_axis_in_bounds(axis, self.ndim)

    def _guard_axis_lock_mutation(self) -> None:
        """Forbid changing per-axis locks while an owner lock is held.

        During an owner lock (e.g. Shapes drawing) the per-axis configuration is
        frozen: the owner's exempt set is authoritative and the padlock UI is
        disabled, so a per-axis mutation would be both surprising and ignored.
        """
        if self._nav_lock_owner is not None:
            raise RuntimeError(
                trans._(
                    'Cannot change per-axis locks while navigation is locked by {owner}.',
                    deferred=True,
                    owner=self._nav_lock_owner,
                )
            )

    def lock_axis(self, axis: int | str) -> None:
        """Lock a single axis so navigation cannot move its slice position.

        A *persistent, user-facing* per-axis lock, distinct from the transient
        ``lock_navigation`` owner lock. While an axis is locked, ``set_point`` /
        ``set_current_step`` (and everything that funnels through them — the
        sliders, slice-number editor, playback, wheel and arrow-key stepping)
        are no-ops for that axis, and ``_focus_up``/``_focus_down`` skip it.

        Enforcement is method-level only (see ``NAVIGATION_LOCK_VERSION``).
        Direct field assignment — ``dims.point = ...``,
        ``dims.current_step = ...`` — and ``set_point(..., force=True)``
        deliberately bypass the lock; that direct path is the intended
        programmatic **escape hatch**. The lock guards deliberate *navigation*
        (the methods and the UI), not raw coordinate writes: the internal
        validator and lifecycle resets (``reset``, ``_go_to_center_step``) also
        assign those fields, so they are not gated.

        While a ``lock_navigation`` owner lock is held the per-axis locks are
        *suspended* (the owner's exempt set governs) and mutating them raises
        ``RuntimeError`` — the per-axis configuration is frozen for the
        duration.

        Parameters
        ----------
        axis : int or str
            Axis index, or an axis label (see ``axis_labels``). A label that
            matches no axis, or more than one, raises ``ValueError``.
        """
        self._guard_axis_lock_mutation()
        ax = self._normalize_axis(axis)
        new = list(self.axis_locked)
        new[ax] = True
        self.axis_locked = tuple(new)

    def unlock_axis(self, axis: int | str) -> None:
        """Unlock a single axis so navigation may move it again.

        The inverse of ``lock_axis``. Operates only on the sticky per-axis lock
        tier; it never touches an owner lock. Raises ``RuntimeError`` while an
        owner lock is held (see ``lock_axis``).

        Parameters
        ----------
        axis : int or str
            Axis index or an axis label (see ``lock_axis`` for name resolution).
        """
        self._guard_axis_lock_mutation()
        ax = self._normalize_axis(axis)
        new = list(self.axis_locked)
        new[ax] = False
        self.axis_locked = tuple(new)

    def lock_all_axes(self) -> None:
        """Lock every axis (see ``lock_axis``).

        Raises ``RuntimeError`` while an owner lock is held.
        """
        self._guard_axis_lock_mutation()
        self.axis_locked = (True,) * self.ndim

    def unlock_all_axes(self) -> None:
        """Unlock every axis (see ``lock_axis``).

        Raises ``RuntimeError`` while an owner lock is held.
        """
        self._guard_axis_lock_mutation()
        self.axis_locked = (False,) * self.ndim

    def _sanitize_input(
        self, axis, value, value_is_sequence=False
    ) -> tuple[list[int], list]:
        """
        Ensure that axis and value are the same length, that axes are not
        out of bounds, and coerces to lists for easier processing.
        """
        if isinstance(axis, Integral):
            if (
                isinstance(value, Sequence)
                and not isinstance(value, str)
                and not value_is_sequence
            ):
                raise ValueError(
                    trans._('cannot set multiple values to a single axis')
                )
            axis = [axis]
            value = [value]
        else:
            axis = list(axis)
            value = list(value)

        if len(axis) != len(value):
            raise ValueError(
                trans._('axis and value sequences must have equal length')
            )

        # Normalize to canonical non-negative indices so downstream comparisons
        # (notably the navigation-lock exempt set) and callers see a single axis
        # numbering rather than a mix of negative and positive indices.
        axis = [ensure_axis_in_bounds(ax, self.ndim) for ax in axis]
        return axis, value

    @contextlib.contextmanager
    def _validating_ctx(self):
        prev = self._validating
        self._validating = True
        try:
            yield
        finally:
            self._validating = prev


def ensure_len(value: tuple, length: int, pad_width: Any):
    """
    Ensure that the value has the required number of elements.

    Right-crop if value is too long; left-pad with default if too short.

    Parameters
    ----------
    value : Tuple
        A tuple of values to be resized.
    ndim : int
        Number of desired values.
    default : Tuple
        Default element for left-padding.
    """
    if len(value) < length:
        # left pad
        value = (pad_width,) * (length - len(value)) + value
    elif len(value) > length:
        # right-crop
        value = value[-length:]
    return value


def ensure_axis_in_bounds(axis: int, ndim: int) -> int:
    """Ensure a given value is inside the existing axes of the image.

    Returns
    -------
    axis : int
        The axis which was checked for validity.
    ndim : int
        The dimensionality of the layer.

    Raises
    ------
    ValueError
        The given axis index is out of bounds.
    """
    if axis not in range(-ndim, ndim):
        msg = trans._(
            'Axis {axis} not defined for dimensionality {ndim}. Must be in [{ndim_lower}, {ndim}).',
            deferred=True,
            axis=axis,
            ndim=ndim,
            ndim_lower=-ndim,
        )
        raise ValueError(msg)

    return axis % ndim
