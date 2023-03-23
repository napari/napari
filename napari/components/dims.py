import warnings
from collections import namedtuple
from numbers import Integral
from typing import (
    Any,
    Literal,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from pydantic import root_validator, validator

from napari.utils.events import EventedModel
from napari.utils.misc import argsort, reorder_after_dim_reduction
from napari.utils.translations import trans

RangeTuple = namedtuple('RangeTuple', ('start', 'stop', 'step'))


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
    point_slider : tuple of int
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
    range: Tuple[RangeTuple[float, float, float], ...] = ()
    left_margin: Tuple[float, ...] = ()
    right_margin: Tuple[float, ...] = ()
    point: Tuple[float, ...] = ()

    last_used: int = 0

    # private vars
    _scroll_progress: int = 0

    # validators
    # check fields is false to allow private fields to work
    @validator(
        'order',
        'axis_labels',
        'point',
        'left_margin',
        'right_margin',
        pre=True,
        allow_reuse=True,
    )
    def _as_tuple(v):
        return tuple(v)

    @validator('range', pre=True)
    def _sorted_ranges(v):
        range_ = []
        for rng in v:
            start, stop = sorted(rng[:2])
            step = rng[2]
            # ensure step is not bigger than full range thickness and coerce to proper type
            range_.append((start, stop, np.clip(step, 0, stop - start)))
        return range_

    @root_validator(skip_on_failure=True, allow_reuse=True)
    def _check_dims(cls, values):
        """Check the consitency of dimensionaity for all attributes

        Parameters
        ----------
        values : dict
            Values dictionary to update dims model with.
        """
        ndim = values['ndim']

        range_ = ensure_ndim(values['range'], ndim, default=(0.0, 2.0, 1.0))
        values['range'] = tuple(RangeTuple(*rng) for rng in range_)

        point = ensure_ndim(values['point'], ndim, default=0.0)
        # ensure point is limited to range
        values['point'] = tuple(
            np.clip(pt, rng.start, rng.stop)
            for pt, rng in zip(point, values['range'])
        )

        # TODO: do we need to coerce the margins at all here or are we ok with
        #       margins going out of the range?
        values['left_margin'] = ensure_ndim(
            values['left_margin'], ndim, default=0.0
        )
        values['right_margin'] = ensure_ndim(
            values['right_margin'], ndim, default=0.0
        )

        # order and label default computation is too different to include in ensure_ndim()
        # Check the order tuple has same number of elements as ndim
        order = values['order']
        order_ndim = len(order)
        if len(order) < ndim:
            # new dims are always prepended
            prepended_dims = tuple(range(ndim - order_ndim))
            # maintain existing order, but shift accordingly
            existing_order = tuple(o + ndim - order_ndim for o in order)
            values['order'] = prepended_dims + existing_order
        elif len(order) > ndim:
            values['order'] = reorder_after_dim_reduction(order[-ndim:])

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
        self.events.point_slider.connect(self.events.current_step)

    @property
    def nsteps(self) -> Tuple[float]:
        return tuple(
            # "or 1" ensures degenerate dimension works
            int((rng.stop - rng.start) / (rng.step or 1))
            for rng in self.range
        )

    @nsteps.setter
    def nsteps(self, value):
        self.range = [
            (rng.start, rng.stop, (rng.stop - rng.start) / nsteps)
            for rng, nsteps in zip(self.range, value)
        ]

    @property
    def current_step(self) -> Tuple[int]:
        warnings.warn(
            trans._(
                'Dims.current_step is deprecated. Use Dims.point_slider instead.'
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.point_slider

    @current_step.setter
    def current_step(self, value: Tuple[int]):
        warnings.warn(
            trans._(
                'Dims.current_step is deprecated. Use Dims.point_slider instead.'
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        self.point_slider = value

    @property
    def point_slider(self):
        return tuple(
            int(round((point - rng.start) / (rng.step or 1)))
            for point, rng in zip(self.point, self.range)
        )

    @point_slider.setter
    def point_slider(self, value):
        self.point = [
            rng.start + point * rng.step
            for point, rng in zip(value, self.range)
        ]

    @property
    def thickness(self) -> Tuple[float]:
        return tuple(
            left + right
            for left, right in zip(self._left_margin, self._right_margin)
        )

    @thickness.setter
    def thickness(self, value):
        self._left_margin = self._right_margin = (
            tuple(val / 2) for val in value
        )

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

    def set_point(
        self,
        axis: Union[int, Sequence[int]],
        value: Union[Union[int, float], Sequence[Union[int, float]]],
    ):
        """Sets point to slice dimension in world coordinates.

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
        full_point = list(self.point)
        for ax, val in zip(axis, value):
            full_point[ax] = val
        self.point = full_point

    def set_point_slider(
        self,
        axis: Union[int, Sequence[int]],
        value: Union[int, Sequence[int]],
    ):
        axis, value = self._sanitize_input(
            axis, value, value_is_sequence=False
        )
        range_ = list(self.range)
        value_world = []
        for ax, val in zip(axis, value):
            rng = range_[ax]
            value_world.append(rng.start + val * rng.step)
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

    def reset(self):
        """Reset dims values to initial states."""
        # Don't reset axis labels
        self.update(
            {
                "range": ((0, 2, 1),) * self.ndim,
                "point": (0,) * self.ndim,
                "order": tuple(range(self.ndim)),
                "left_margin": (0,) * self.ndim,
                "right_margin": (0,) * self.ndim,
            }
        )

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
        self.set_point_slider(axis, self.point_slider[axis] + 1)

    def _increment_dims_left(self, axis: int = None):
        """Increment dimensions to the left along given axis, or last used axis if None

        Parameters
        ----------
        axis : int, optional
            Axis along which to increment dims, by default None
        """
        if axis is None:
            axis = self.last_used
        self.set_point_slider(axis, self.point_slider[axis] - 1)

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


def ensure_ndim(value: Tuple, ndim: int, default: Any):
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
