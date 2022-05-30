import warnings
from typing import List, Literal

import numpy as np
from pydantic import root_validator, validator

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

    # fields (order matters for validation! Previous fields are available for following fields)
    ndim: int = 2
    ndisplay: Literal[2, 3] = 2
    axis_labels: EventedList[str] = ['0', '1']
    range: NestableEventedList[NestableEventedList[float]] = [[0, 2], [0, 2]]
    span: NestableEventedList[NestableEventedList[float]] = [[0, 0], [0, 0]]
    step: EventedList[float] = [1, 1]
    order: EventedList[int] = [0, 1]
    last_used: int = 0
    # private vars
    _scroll_progress: int = 0

    class Config:
        computed_fields = {
            '_span_step': ['span', 'range', 'step'],
            'nsteps': ['range', 'step'],
            'thickness': ['span'],
            '_thickness_step': ['span', 'step'],
            'point': ['span'],
            '_point_step': ['span', 'range', 'step'],
            'current_step': ['span', 'range', 'step'],
            'displayed': ['order', 'ndisplay'],
            'not_displayed': ['order', 'ndisplay'],
            'displayed_order': ['order', 'ndisplay'],
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # to be removed if/when deprecating current_step
        self.events.span.connect(self.events.current_step)

    @validator('axis_labels', pre=True, always=True)
    def _listify_string(v):
        return list(v)

    @root_validator(skip_on_failure=True)
    def _enforce_ndim(cls, values):
        # This validator should not be split, or setting attributes will only trigger
        # invididual validators, potentially not updating the other fields accordingly
        ndim = values['ndim']

        # axis labels
        labels = list(values['axis_labels'])
        if len(labels) < ndim:
            # Append new "default" labels to existing ones
            if labels == list(map(str, range(len(labels)))):
                labels = list(map(str, range(ndim)))
            else:
                labels = list(map(str, range(ndim - len(labels)))) + labels
        elif len(labels) > ndim:
            labels = labels[-ndim:]

        # range
        range_ = ensure_ndim(values['range'], ndim, default=[0, 2])
        range_ = [sorted(v) for v in range_]

        # span
        span = ensure_ndim(values['span'], ndim, default=[0, 0])
        span = [sorted(v) for v in span]
        # ensure span is limited to range
        for i, ((low, high), (min_val, max_val)) in enumerate(
            zip(span, range_)
        ):
            low = np.clip(low, min_val, max_val)
            high = np.clip(high, min_val, max_val)
            span[i] = [low, high]

        # step
        step = ensure_ndim(values['step'], ndim, default=1)
        # ensure step is not bigger than range
        for i, (stp, (min_val, max_val)) in enumerate(
            zip(
                step,
                range_,
            )
        ):
            step[i] = np.clip(stp, 0, max_val - min_val)

        order = values['order']
        if len(order) < ndim:
            order = list(range(ndim - len(order))) + [
                o + ndim - len(order) for o in order
            ]
        elif len(order) > ndim:
            order = reorder_after_dim_reduction(order[-ndim:])

        # Check the order is a permutation of 0, ..., ndim - 1
        if not set(order) == set(range(ndim)):
            raise ValueError(
                trans._(
                    "Invalid ordering {order} for {ndim} dimensions",
                    deferred=True,
                    order=order,
                    ndim=ndim,
                )
            )

        values.update(
            {
                'axis_labels': labels,
                'range': range_,
                'span': span,
                'step': step,
                'order': order,
            }
        )
        return values

    @property
    def _span_step(self) -> List[float]:
        return [
            [
                int(round((low - min_val) / step)),
                int(round((high - min_val) / step)),
            ]
            for (low, high), (min_val, _), step in zip(
                self.span, self.range, self.step
            )
        ]

    @_span_step.setter
    def _span_step(self, value):
        self.span = [
            [
                min_val + low * step,
                min_val + high * step,
            ]
            for (low, high), (min_val, _), step in zip(
                value, self.range, self.step
            )
        ]

    @property
    def nsteps(self) -> List[float]:
        return [
            int((max_val - min_val) // step)
            for (min_val, max_val), step in zip(self.range, self.step)
        ]

    @nsteps.setter
    def nsteps(self, value):
        self.step = [
            (max_val - min_val) / nsteps
            for (min_val, max_val), nsteps in zip(self.range, value)
        ]

    @property
    def thickness(self) -> List[float]:
        return [high - low for low, high in self.span]

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
            span.append([new_low, new_high])
        self.span = span

    @property
    def _thickness_step(self) -> List[float]:
        return [
            thickness / step
            for thickness, step in zip(self.thickness, self.step)
        ]

    @_thickness_step.setter
    def _thickness_step(self, value):
        self.thickness = [
            thickness * step for thickness, step in zip(value, self.step)
        ]

    @property
    def point(self) -> List[float]:
        return [(low + high) / 2 for low, high in self.span]

    @point.setter
    def point(self, value):
        span = []
        for point, (min_val, max_val), thickness in zip(
            value, self.range, self.thickness
        ):
            half_thk = thickness / 2
            min_pt, max_pt = (min_val + half_thk, max_val - half_thk)
            point = np.clip(point, min_pt, max_pt)
            span.append([point - half_thk, point + half_thk])
        self.span = span

    @property
    def _point_step(self):
        return [
            int(round((point - min_val) / step))
            for point, (min_val, _), step in zip(
                self.point, self.range, self.step
            )
        ]

    @_point_step.setter
    def _point_step(self, value):
        self.point = [
            min_val + point * step
            for point, (min_val, _), step in zip(value, self.range, self.step)
        ]

    @property
    def current_step(self) -> List[int]:
        warnings.warn(
            trans._(
                'Dims.current_step is deprecated. Use Dims._point_step instead.'
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return self._point_step

    @current_step.setter
    def current_step(self, value: List[int]):
        warnings.warn(
            trans._(
                'Dims.current_step is deprecated. Use Dims._point_step instead.'
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        self._point_step = value

    @property
    def displayed(self) -> List[int]:
        """list: Dimensions that are displayed."""
        return self.order[-self.ndisplay :]

    @displayed.setter
    def displayed(self, value):
        self.order[-self.ndisplay :] = value

    @property
    def not_displayed(self) -> List[int]:
        """list: Dimensions that are not displayed."""
        return self.order[: -self.ndisplay]

    @not_displayed.setter
    def not_displayed(self, value):
        self.order[: -self.ndisplay] = value

    @property
    def displayed_order(self) -> List[int]:
        displayed = self.displayed
        # equivalent to: order = np.argsort(self.displayed)
        order = sorted(range(len(displayed)), key=lambda x: displayed[x])
        return list(order)

    def reset(self):
        """Reset dims values to initial states."""
        # Don't reset axis labels
        # TODO: here and in other places, we could use `self.update` to reduce validations
        # the downside is that (currently) `self.update` does not trigger events for the
        # individual fields.
        self.range = [[0, 2]] * self.ndim
        self.span = [[0, 0]] * self.ndim
        self.step = [1] * self.ndim
        self.order = list(range(self.ndim))

    def _increment_dims_right(self, axis: int = None):
        """Increment dimensions to the right along given axis, or last used axis if None

        Parameters
        ----------
        axis : int, optional
            Axis along which to increment dims, by default None
        """
        if axis is None:
            axis = self.last_used
        self._point_step[axis] += 1

    def _increment_dims_left(self, axis: int = None):
        """Increment dimensions to the left along given axis, or last used axis if None

        Parameters
        ----------
        axis : int, optional
            Axis along which to increment dims, by default None
        """
        if axis is None:
            axis = self.last_used
        self._point_step[axis] -= 1

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
        """Transpose last two displayed dimensions."""
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
    return list(arr)


def ensure_ndim(value, ndim, default):
    """Ensure that the value has same number of elements as ndim"""
    if len(value) < ndim:
        # left pad
        value = [default] * (ndim - len(value)) + value
    elif len(value) > ndim:
        # right-crop
        value = value[-ndim:]
    return value
