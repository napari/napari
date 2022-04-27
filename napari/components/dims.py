import warnings
from typing import Sequence, Tuple, Union

import numpy as np
from pydantic import root_validator, validator
from typing_extensions import Literal  # Added to typing in 3.8

from ..utils.events import EventedList, EventedModel, NestableEventedList
from ..utils.translations import trans


class Dims(EventedModel):
    """Dimensions object modeling slicing and displaying.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    ndisplay : int
        Number of displayed dimensions.
    last_used : int
        Dimension which was last used.
    range : tuple of 3-tuple of float
        List of tuples (min, max, step), one for each dimension in world
        coordinates space.
    span : tuple of 3-tuple of float
        Tuple of (low, high) bounds of the currently selected slice in world space.
    order : tuple of int
        Tuple of ordering the dimensions, where the last dimensions are rendered.
    axis_labels : tuple of str
        Tuple of labels for each dimension.

    Attributes
    ----------
    ndim : int
        Number of dimensions.
    ndisplay : int
        Number of displayed dimensions.
    last_used : int
        Dimension which was last used.
    range : tuple of 3-tuple of float
        List of tuples (min, max, step), one for each dimension in world
        coordinates space.
    span : tuple of 3-tuple of float
        Tuple of (low, high) bounds of the currently selected slice in world space.
    order : tuple of int
        Tuple of ordering the dimensions, where the last dimensions are rendered.
    axis_labels : tuple of str
        Tuple of labels for each dimension.
    current_step : tuple of int
        Tuple the slider position for each dims slider, in world coordinates.
    nsteps : tuple of int
        Number of steps available to each slider. These are calculated from
        the ``range``.
    thickness : tuple of floats
        Thickness of each span in world coordinates.
    point : tuple of floats
        Center point of each span in world coordinates.
    displayed : tuple of int
        List of dimensions that are displayed. These are calculated from the
        ``order`` and ``ndisplay``.
    not_displayed : tuple of int
        List of dimensions that are not displayed. These are calculated from the
        ``order`` and ``ndisplay``.
    displayed_order : tuple of int
        Order of only displayed dimensions. These are calculated from the
        ``displayed`` dimensions.
    """

    # fields
    ndim: int = 2
    ndisplay: Literal[2, 3] = 2
    order: EventedList[int] = ()
    axis_labels: EventedList[str] = ()
    range: EventedList[EventedList[float, float]] = ()
    span: EventedList[EventedList[float, float]] = ()
    step: EventedList[float] = ()
    last_used: int = 0

    # private vars
    _scroll_progress: int = 0

    # validators
    @validator('axis_labels', pre=True)
    def _string_to_list(v):
        if isinstance(v, str):
            return list(v)
        return v

    @validator('range', 'span', pre=True)
    def _sort_values(v):
        return [sorted(d) for d in v]

    @root_validator
    def _check_dims(cls, values):
        """Check the consitency of dimensionaity for all attributes

        Parameters
        ----------
        values : dict
            Values dictionary to update dims model with.
        """
        ndim = values['ndim']

        values['range'] = ensure_ndim(
            values['range'], ndim, default=(0.0, 2.0)
        )

        values['span'] = ensure_ndim(values['span'], ndim, default=(0.0, 0.0))
        # ensure span is limited to range
        for (low, high), (min_val, max_val) in zip(
            values['span'], values['range']
        ):
            if low < min_val or high > max_val:
                raise ValueError(
                    trans._(
                        "Invalid span {span} for dimension with range {range}",
                        deferred=True,
                        span=(low, high),
                        range=(min_val, max_val),
                    )
                )

        values['step'] = ensure_ndim(values['step'], ndim, default=1.0)
        # ensure step is not bigger than range
        for step, (min_val, max_val) in zip(values['step'], values['range']):
            if step > max_val - min_val:
                raise ValueError(
                    trans._(
                        "Invalid step {step} for dimension with range {range}",
                        deferred=True,
                        step=step,
                        range=(min_val, max_val),
                    )
                )

        # order and label defailt computation is too weird to include in ensure_ndim()
        # Check the order tuple has same number of elements as ndim
        if len(values['order']) < ndim:
            values['order'] = tuple(
                range(ndim - len(values['order']))
            ) + tuple(o + ndim - len(values['order']) for o in values['order'])
        elif len(values['order']) > ndim:
            values['order'] = reorder_after_dim_reduction(
                values['order'][-ndim:]
            )

        # Check the order is a permutation of 0, ..., ndim - 1
        if not set(values['order']) == set(range(ndim)):
            raise ValueError(
                trans._(
                    "Invalid ordering {order} for {ndim} dimensions",
                    deferred=True,
                    order=values['order'],
                    ndim=ndim,
                )
            )

        # Check the axis labels tuple has same number of elements as ndim
        if len(values['axis_labels']) < ndim:
            # Append new "default" labels to existing ones
            if values['axis_labels'] == tuple(
                map(str, range(len(values['axis_labels'])))
            ):
                values['axis_labels'] = tuple(map(str, range(ndim)))
            else:
                values['axis_labels'] = (
                    tuple(map(str, range(ndim - len(values['axis_labels']))))
                    + values['axis_labels']
                )
        elif len(values['axis_labels']) > ndim:
            values['axis_labels'] = values['axis_labels'][-ndim:]

        return values

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # to be removed if/when deprecating current_step
        self.events.span.connect(self.events.current_step)

    @property
    def _span_step(self) -> NestableEventedList[float]:
        return NestableEventedList(
            (
                int(round((low - min_val) / step)),
                int(round((high - min_val) / step)),
            )
            for (low, high), (min_val, _), step in zip(
                self.span, self.range, self.step
            )
        )

    @_span_step.setter
    def _span_step(self, value):
        self.step = [
            min_val + step * val
            for (min_val, _), step, val in zip(self.range, self.step, value)
        ]

    @property
    def nsteps(self) -> EventedList[float]:
        return EventedList(
            int((max_val - min_val) // step)
            for (min_val, max_val), step in zip(self.range, self.step)
        )

    @nsteps.setter
    def nsteps(self, value):
        self.step = [
            (max_val - min_val) / nsteps
            for (min_val, max_val), nsteps in zip(self.range, value)
        ]

    @property
    def thickness(self) -> EventedList[float]:
        return EventedList(high - low for low, high in self.span)

    @thickness.setter
    def thickness(self, value):
        span = []
        for thickness, (min_val, max_val), (low, high) in zip(
            value, self.range, self.span
        ):
            max_change = min(abs(low - min_val), abs(max_val - high))
            thickness_change = min((thickness - (high - low)) / 2, max_change)
            new_low = max(min_val, low - thickness_change)
            new_high = min(max_val, high + thickness_change)
            span.append(new_low, new_high)
        self.span = span

    @property
    def _thickness_step(self) -> EventedList[float]:
        return EventedList(
            int(round((high - low) / step))
            for (high, low), step in zip(self.range, self.step)
        )

    @_thickness_step.setter
    def _thickness_step(self, value):
        self.step = [
            min_val + step * val
            for (min_val, _), step, val in zip(self.range, self.step, value)
        ]

    @property
    def point(self) -> EventedList[float]:
        return EventedList((low + high) / 2 for low, high in self.span)

    @point.setter
    def point(self, value):
        span = []
        for point, (min_val, max_val), thickness in zip(
            value, self.range, self.thickness
        ):
            half_thk = thickness / 2
            min_pt, max_pt = (min_val + half_thk, max_val - half_thk)
            point = np.clip(point, min_pt, max_pt)
            span.append(point - half_thk, point + half_thk)
        self.span = span

    @property
    def current_step(self) -> EventedList[int]:
        return EventedList(
            int(round((point - min_val) / step))
            for point, (min_val, _, step) in zip(self.point, self.range)
        )

    @current_step.setter
    def current_step(self, value: Tuple[int, ...]):
        warnings.warn(
            trans._(
                'Dims.current_step is deprecated. Use Dims.set_point_step instead.'
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        self._point_step = value

    @property
    def displayed(self) -> Tuple[int, ...]:
        """Tuple: Dimensions that are displayed."""
        return self.order[-self.ndisplay :]

    @property
    def not_displayed(self) -> Tuple[int, ...]:
        """Tuple: Dimensions that are not displayed."""
        return self.order[: -self.ndisplay]

    @property
    def displayed_order(self) -> Tuple[int, ...]:
        displayed = self.displayed
        # equivalent to: order = np.argsort(self.displayed)
        order = sorted(range(len(displayed)), key=lambda x: displayed[x])
        return tuple(order)

    def set_span_step(
        self,
        axis: Union[int, Sequence[int]],
        value: Union[Sequence[int], Sequence[Sequence[int]]],
    ):
        axis, value = self._sanitize_input(axis, value, value_is_sequence=True)
        range = list(self.range)
        value_world = []
        for ax, val in zip(axis, value):
            min_val, _, step_size = range[ax]
            value_world.append([min_val + v * step_size for v in val])
        self.set_span(axis, value_world)

    def reset(self):
        """Reset dims values to initial states."""
        # Don't reset axis labels
        self.range = ((0, 2, 1),) * self.ndim
        self.span = ((0, 0),) * self.ndim
        self.order = tuple(range(self.ndim))

    def _increment_dims_right(self, axis: int = None):
        """Increment dimensions to the right along given axis, or last used axis if None

        Parameters
        ----------
        axis : int, optional
            Axis along which to increment dims, by default None
        """
        if axis is None:
            axis = self.last_used
        self.set_point_step(axis, self.current_step[axis] + 1)

    def _increment_dims_left(self, axis: int = None):
        """Increment dimensions to the left along given axis, or last used axis if None

        Parameters
        ----------
        axis : int, optional
            Axis along which to increment dims, by default None
        """
        if axis is None:
            axis = self.last_used
        self.set_point_step(axis, self.current_step[axis] - 1)

    def _focus_up(self):
        """Shift focused dimension slider to be the next slider above."""
        sliders = [d for d in self.not_displayed if self.nsteps[d] > 1]
        if len(sliders) == 0:
            return

        index = (sliders.index(self.last_used) + 1) % len(sliders)
        self.last_used = sliders[index]

    def _focus_down(self):
        """Shift focused dimension slider to be the next slider bellow."""
        sliders = [d for d in self.not_displayed if self.nsteps[d] > 1]
        if len(sliders) == 0:
            return

        index = (sliders.index(self.last_used) - 1) % len(sliders)
        self.last_used = sliders[index]

    def _roll(self):
        """Roll order of dimensions for display."""
        order = np.array(self.order)
        nsteps = np.array(self.nsteps)
        order[nsteps > 1] = np.roll(order[nsteps > 1], 1)
        self.order = order.tolist()

    def _transpose(self):
        """Transpose displayed dimensions."""
        order = list(self.order)
        order[-2], order[-1] = order[-1], order[-2]
        self.order = order


def reorder_after_dim_reduction(order):
    """Ensure current dimension order is preserved after dims are dropped.

    Parameters
    ----------
    order : tuple
        The data to reorder.

    Returns
    -------
    arr : tuple
        The original array with the unneeded dimension
        thrown away.
    """
    arr = sorted(range(len(order)), key=lambda x: order[x])
    return tuple(arr)


def ensure_ndim(value, ndim, default):
    """Ensure that the value has same number of elements as ndim"""
    if len(value) < ndim:
        # left pad
        value = (default,) * (ndim - len(value)) + value
    elif len(value) > ndim:
        # right-crop
        value = value[-ndim:]
    return value
