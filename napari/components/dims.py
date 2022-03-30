from numbers import Integral
from typing import Sequence, Tuple, Union

import numpy as np
from pydantic import root_validator, validator
from typing_extensions import Literal  # Added to typing in 3.8

from ..utils.events import EventedModel
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
        List of tuples (min, max, step), one for each dimension. In a world
        coordinates space. As with Python's `range` and `slice`, max is not
        included.
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
        List of tuples (min, max, step), one for each dimension. In a world
        coordinates space. As with Python's `range` and `slice`, max is not
        included.
    current_step : tuple of int
        Tuple the slider position for each dims slider, in world coordinates.
    order : tuple of int
        Tuple of ordering the dimensions, where the last dimensions are rendered.
    axis_labels : tuple of str
        Tuple of labels for each dimension.
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
    last_used: int = 0
    range: Tuple[Tuple[float, float, float], ...] = ()
    span: Tuple[Tuple[int, int], ...] = ()
    order: Tuple[int, ...] = ()
    axis_labels: Tuple[str, ...] = ()

    # private vars
    _scroll_progress: int = 0

    # validators
    @validator('axis_labels', pre=True)
    def _string_to_list(v):
        if isinstance(v, str):
            return list(v)
        return v

    @root_validator
    def _check_dims(cls, values):
        """Check the consitency of dimensionaity for all attributes

        Parameters
        ----------
        values : dict
            Values dictionary to update dims model with.
        """
        ndim = values['ndim']

        # Check the range tuple has same number of elements as ndim
        if len(values['range']) < ndim:
            values['range'] = ((0, 2, 1),) * (
                ndim - len(values['range'])
            ) + values['range']
        elif len(values['range']) > ndim:
            values['range'] = values['range'][-ndim:]

        # Check the span tuple has same number of elements as ndim
        if len(values['span']) < ndim:
            values['span'] = ((0, 0),) * (ndim - len(values['span'])) + values[
                'span'
            ]
        elif len(values['span']) > ndim:
            values['span'] = values['span'][-ndim:]

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

    @property
    def nsteps(self) -> Tuple[int, ...]:
        """Tuple of int: Number of slider steps for each dimension."""
        return tuple(
            int((max_val - min_val) // step_size)
            for min_val, max_val, step_size in self.range
        )

    @property
    def thickness(self) -> Tuple[int]:
        return tuple(high - low for low, high in self.span)

    @property
    def point(self) -> Tuple[float, ...]:
        return tuple((low + high) / 2 for low, high in self.span)

    @property
    def current_step(self) -> Tuple[int, ...]:
        return tuple(
            round((point - min_val) / step)
            for point, (min_val, _, step) in zip(self.point, self.range)
        )

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

    def set_range(
        self,
        axis: Union[int, Sequence[int]],
        _range: Union[
            Sequence[Union[int, float]], Sequence[Sequence[Union[int, float]]]
        ],
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
        for ax, val in zip(axis, value):
            full_range[ax] = val
        self.range = full_range

    def set_span(
        self,
        axis: Union[int, Sequence[int]],
        value: Union[
            Sequence[Union[int, float]], Sequence[Sequence[Union[int, float]]]
        ],
    ):
        axis, value = self._sanitize_input(axis, value, value_is_sequence=True)
        full_span = list(self.span)
        range = list(self.range)
        for ax, val in zip(axis, value):
            min_val, max_val, _ = range[ax]
            low, high = sorted(val)
            span = max(min_val, low), min(max_val, high)
            full_span[ax] = span
        self.span = full_span

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

    def set_point(
        self,
        axis: Union[int, Sequence[int]],
        value: Union[Union[int, float], Sequence[Union[int, float]]],
    ):
        """Sets point to slice dimension in world coordinates.

        The current thickness is preserved, and the point is set as
        the center of the slice.

        Parameters
        ----------
        axis : int or sequence of int
            Dimension index or a sequence of axes whos point will be set.
        value : scalar or sequence of scalars
            Value of the point for each axis.
        """
        axis, value = self._sanitize_input(
            axis, value, value_is_sequence=False
        )
        full_span = list(self.span)
        point = list(self.point)
        range = list(self.range)
        for ax, val in zip(axis, value):
            shift = val - point[ax]
            min_val, max_val, _ = range[ax]
            low, high = tuple(v + shift for v in full_span[ax])
            span = max(min_val, low), min(max_val, high)
            full_span[ax] = span
        self.span = full_span

    def set_point_step(
        self,
        axis: Union[int, Sequence[int]],
        value: Union[Union[int, float], Sequence[Union[int, float]]],
    ):
        axis, value = self._sanitize_input(
            axis, value, value_is_sequence=False
        )
        range = list(self.range)
        value_world = []
        for ax, val in zip(axis, value):
            min_val, _, step_size = range[ax]
            value_world.append(min_val + val * step_size)
        self.set_point(axis, value_world)

    def set_axis_label(
        self,
        axis: Union[int, Sequence[int]],
        label: Union[str, Sequence[str]],
    ):
        """Sets new axis labels for the given axes.

        Parameters
        ----------
        axis : int or sequence of int
            Dimension index or a sequence of axes whos labels will be set.
        label : str or sequence of str
            Given labels for the specified axes.
        """
        axis, value = self._sanitize_input(
            axis, label, value_is_sequence=False
        )
        full_axis_labels = list(self.axis_labels)
        for ax, val in zip(axis, label):
            full_axis_labels[ax] = val
        self.axis_labels = full_axis_labels

    def set_thickness(
        self,
        axis: Union[int, Sequence[int]],
        value: Union[Union[int, float], Sequence[Union[int, float]]],
    ):
        """Set the slider slice thickness for this dimension.

        Parameters
        ----------
        axis : int or sequence of int
            Dimension index or a sequence of axes whose slice thickness will be set.
        value : scalar or sequence of scalars
            Value of the slice thickness.
        """
        axis, value = self._sanitize_input(
            axis, value, value_is_sequence=False
        )
        full_span = list(self.span)
        range = list(self.range)
        for ax, val in zip(axis, value):
            min_val, max_val, _ = range[ax]
            low, high = full_span[ax]
            thickness_change = (val - (high - low)) / 2
            new_low = max(min_val, low - thickness_change)
            new_high = min(max_val, high + thickness_change)
            full_span[ax] = new_low, new_high
        self.span = full_span

    def set_thickness_step(
        self,
        axis: Union[int, Sequence[int]],
        value: Union[int, Sequence[int]],
    ):
        axis, value = self._sanitize_input(
            axis, value, value_is_sequence=False
        )
        range = list(self.range)
        value_world = []
        for ax, val in zip(axis, value):
            min_val, _, step_size = range[ax]
            value_world.append(min_val + val * step_size)
        self.set_thickness(axis, value_world)

    def reset(self):
        """Reset dims values to initial states."""
        # Don't reset axis labels
        self.range = ((0, 2, 1),) * self.ndim
        self.current_step = (0,) * self.ndim
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
        self.set_current_step(axis, self.current_step[axis] + 1)

    def _increment_dims_left(self, axis: int = None):
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

    def _sanitize_input(self, axis, value, value_is_sequence=False):
        """
        Ensure that axis is and value are the same length, that axis are not
        out of bounds, and corerces to lists for easier processing.
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
                trans._("axis and value sequences must have equal length")
            )

        for ax in axis:
            assert_axis_in_bounds(ax, self.ndim)
        return axis, value


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


def assert_axis_in_bounds(axis: int, ndim: int) -> int:
    """Assert a given value is inside the existing axes of the image.

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
