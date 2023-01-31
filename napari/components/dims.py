import warnings
from numbers import Integral
from typing import Literal  # Added to typing in 3.8
from typing import Sequence, Tuple, Union

import numpy as np
from pydantic import root_validator, validator

from napari.utils.events import EventedModel
from napari.utils.misc import argsort, reorder_after_dim_reduction
from napari.utils.translations import trans


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
    point_step : tuple of int
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
    order: Tuple[int, ...] = ()
    axis_labels: Tuple[str, ...] = ()
    range: Tuple[Tuple[float, float], ...] = ()
    step: Tuple[float, ...] = ()
    span: Tuple[Tuple[float, float], ...] = ()

    last_used: int = 0

    # private vars
    _scroll_progress: int = 0

    # validators
    @validator('order', 'axis_labels', 'step', pre=True)
    def _as_tuple(v):
        return tuple(v)

    @validator('range', 'span', pre=True)
    def _sorted_ranges(v):
        return tuple(sorted(float(el) for el in dim) for dim in v)

    @root_validator(skip_on_failure=True)
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

        # order and label default computation is too different to include in ensure_ndim()
        # Check the order tuple has same number of elements as ndim
        order_ndim = len(values['order'])
        if order_ndim < ndim:
            prepended_dims = tuple(
                dim for dim in range(ndim) if dim not in values['order']
            )
            values['order'] = prepended_dims + values['order']
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
        labels_ndim = len(values['axis_labels'])
        if labels_ndim < ndim:
            # Append new "default" labels to existing ones
            if values['axis_labels'] == tuple(map(str, range(labels_ndim))):
                values['axis_labels'] = tuple(map(str, range(ndim)))
            else:
                values['axis_labels'] = (
                    tuple(map(str, range(ndim - labels_ndim)))
                    + values['axis_labels']
                )
        elif labels_ndim > ndim:
            values['axis_labels'] = values['axis_labels'][-ndim:]

        return values

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # to be removed if/when deprecating current_step
        self.events.span.connect(self.events.current_step)

    @property
    def _span_step(self) -> Tuple[float, ...]:
        return tuple(
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
        self.span = [
            (
                min_val + low * step,
                min_val + high * step,
            )
            for (low, high), (min_val, _), step in zip(
                value, self.range, self.step
            )
        ]

    @property
    def nsteps(self) -> Tuple[float]:
        return tuple(
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
    def thickness(self) -> Tuple[float]:
        return tuple(high - low for low, high in self.span)

    @thickness.setter
    def thickness(self, value):
        # change slice thickness by resizing the span symmetrically
        span = []
        for new_thickness, (min_val, max_val), (low, high) in zip(
            value, self.range, self.span
        ):
            # find the maximum possible change in thickness (can't go further than range)
            max_change = min(abs(low - min_val), abs(max_val - high))
            # move low and high end of the span by half the thickness change
            thickness_change = min(
                (new_thickness - (high - low)) / 2, max_change
            )
            new_low = max(min_val, low - thickness_change)
            new_high = min(max_val, high + thickness_change)
            span.append((new_low, new_high))
        self.span = span

    @property
    def _thickness_step(self) -> Tuple[float]:
        return tuple(
            thickness / step
            for thickness, step in zip(self.thickness, self.step)
        )

    @_thickness_step.setter
    def _thickness_step(self, value):
        self.thickness = [
            thickness * step for thickness, step in zip(value, self.step)
        ]

    @property
    def point(self) -> Tuple[float]:
        return tuple((low + high) / 2 for low, high in self.span)

    @point.setter
    def point(self, value):
        # move the slice so its center is on the specified point, preserving thickness
        # if not possible, move it as far as it can be moved
        span = []
        for point, (min_val, max_val), thickness in zip(
            value, self.range, self.thickness
        ):
            # calculate real limits, including half thickness
            half_thk = thickness / 2
            min_pt, max_pt = (min_val + half_thk, max_val - half_thk)
            point = np.clip(point, min_pt, max_pt)
            span.append((point - half_thk, point + half_thk))
        self.span = span

    @property
    def point_step(self):
        return tuple(
            int(round((point - min_val) / step))
            for point, (min_val, _), step in zip(
                self.point, self.range, self.step
            )
        )

    @point_step.setter
    def point_step(self, value):
        self.point = [
            min_val + point * step
            for point, (min_val, _), step in zip(value, self.range, self.step)
        ]

    @property
    def current_step(self) -> Tuple[int]:
        warnings.warn(
            trans._(
                'Dims.current_step is deprecated. Use Dims.point_step instead.'
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.point_step

    @current_step.setter
    def current_step(self, value: Tuple[int]):
        warnings.warn(
            trans._(
                'Dims.current_step is deprecated. Use Dims.point_step instead.'
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        self.point_step = value

    @property
    def displayed(self) -> Tuple[int]:
        """Tuple: Dimensions that are displayed."""
        return self.order[-self.ndisplay :]

    @displayed.setter
    def displayed(self, value):
        self.order = value

    @property
    def not_displayed(self) -> Tuple[int]:
        """Tuple: Dimensions that are not displayed."""
        return self.order[: -self.ndisplay]

    @property
    def displayed_order(self) -> Tuple[int]:
        return tuple(argsort(self.displayed))

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

    def _set_span(
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
            min_val, max_val = range[ax]
            low, high = sorted(val)
            span = max(min_val, low), min(max_val, high)
            full_span[ax] = span
        self.span = full_span

    def _set_span_step(
        self,
        axis: Union[int, Sequence[int]],
        value: Union[Sequence[int], Sequence[Sequence[int]]],
    ):
        axis, value = self._sanitize_input(axis, value, value_is_sequence=True)
        range = list(self.range)
        step = list(self.step)
        value_world = []
        for ax, val in zip(axis, value):
            min_val, _ = range[ax]
            step_size = step[ax]
            value_world.append([min_val + v * step_size for v in val])
        self.set_span(axis, value_world)

    def set_point(
        self,
        axis: Union[int, Sequence[int]],
        value: Union[Union[int, float], Sequence[Union[int, float]]],
    ):
        """Sets point to slice dimension in world coordinates.

        The current thickness is preserved, and the point is set as
        the center of the slice. If too close to (or beyond) the edge to fit
        the whole thickness, the position is clipped as appropriate.

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
        point_clips = [
            (mn + th / 2, mx - th / 2)
            for (mn, mx), th in zip(range, self.thickness)
        ]
        for ax, val in zip(axis, value):
            val = np.clip(val, *point_clips[ax])
            shift = val - point[ax]
            min_val, max_val = range[ax]
            low, high = tuple(v + shift for v in full_span[ax])
            span = max(min_val, low), min(max_val, high)
            full_span[ax] = span
        self.span = full_span

    def set_point_step(
        self,
        axis: Union[int, Sequence[int]],
        value: Union[int, Sequence[int]],
    ):
        axis, value = self._sanitize_input(
            axis, value, value_is_sequence=False
        )
        range = list(self.range)
        step = list(self.step)
        value_world = []
        for ax, val in zip(axis, value):
            min_val, _ = range[ax]
            step_size = step[ax]
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
        axis, label = self._sanitize_input(
            axis, label, value_is_sequence=False
        )
        full_axis_labels = list(self.axis_labels)
        for ax, val in zip(axis, label):
            full_axis_labels[ax] = val
        self.axis_labels = full_axis_labels

    def _set_thickness(
        self,
        axis: Union[int, Sequence[int]],
        value: Union[Union[int, float], Sequence[Union[int, float]]],
    ):
        """Set the slider slice thickness for this dimension. If the new thickness
        would extend beyond the range limits, it is instead clipped to prevent it.

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
            min_val, max_val = range[ax]
            low, high = full_span[ax]
            max_change = min(abs(low - min_val), abs(max_val - high))
            thickness_change = min((val - (high - low)) / 2, max_change)
            new_low = max(min_val, low - thickness_change)
            new_high = min(max_val, high + thickness_change)
            full_span[ax] = new_low, new_high
        self.span = full_span

    def _set_thickness_step(
        self,
        axis: Union[int, Sequence[int]],
        value: Union[int, Sequence[int]],
    ):
        axis, value = self._sanitize_input(
            axis, value, value_is_sequence=False
        )
        range = list(self.range)
        step = list(self.step)
        value_world = []
        for ax, val in zip(axis, value):
            min_val, _ = range[ax]
            step_size = step[ax]
            value_world.append(min_val + val * step_size)
        self.set_thickness(axis, value_world)

    def reset(self):
        """Reset dims values to initial states."""
        # Don't reset axis labels
        self.range = ((0, 2),) * self.ndim
        self.step = (1,) * self.ndim
        self.span = ((0, 0),) * self.ndim
        self.order = tuple(range(self.ndim))

    def transpose(self):
        """Transpose displayed dimensions.

        This swaps the order of the last two displayed dimensions.
        The order of the displayed is taken from Dims.order.
        """
        order = list(self.order)
        order[-2], order[-1] = order[-1], order[-2]
        self.order = order

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
            ensure_axis_in_bounds(ax, self.ndim)
        return axis, value


def ensure_ndim(value: Tuple, ndim: int, default: Tuple):
    """
    Ensure that the value has same number of elements as ndim.

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
    if len(value) < ndim:
        # left pad
        value = (default,) * (ndim - len(value)) + value
    elif len(value) > ndim:
        # right-crop
        value = value[-ndim:]
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
