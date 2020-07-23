import warnings
from typing import Union, Sequence
import numpy as np

from ..utils.events import EventedList
from ..utils.dataclass import dataclass, Property
from dataclasses import field


def only_2D_3D(ndisplay):
    if ndisplay not in (2, 3):
        raise ValueError(
            f"Invalid number of dimensions to be displayed {ndisplay}"
            f" must be either 2 or 3."
        )
    else:
        return ndisplay


def reorder_after_dim_reduction(order):
    """Ensure current dimension order is preserved after dims are dropped.

    Parameters
    ----------
    order : list-like
        The data to reorder.

    Returns
    -------
    arr : list
        The original array with the unneeded dimension
        thrown away.
    """
    arr = np.array(order)
    arr[np.argsort(arr)] = range(len(arr))
    return arr.tolist()


@dataclass(events=True, properties=True)
class Dims:
    """Dimensions object modeling slicing and displaying.

    Parameters
    ----------
    ndim : int, optional
        Number of dimensions.
    ndisplay : int, optional
        Number of displayed dimensions.
    order : list of int, optional
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    axis_labels : list of str, optional
        Dimension names.

    Attributes
    ----------
    events : EmitterGroup
        Event emitter group
    range : list of 3-tuple
        List of tuples (min, max, step), one for each dimension.
    point : list of float
        List of floats setting the current value of the range slider, one for
        each dimension.
    clip : bool
        Flag if to clip indices based on range. Needed for image-like
        layers, but prevents shape-like layers from adding new shapes
        outside their range.
    ndim : int
        Number of dimensions.
    indices : tuple of slice object
        Tuple of slice objects for slicing arrays on each dimension, one for
        each dimension.
    displayed : tuple
        List of dimensions that are displayed.
    not_displayed : tuple
        List of dimensions that are not displayed.
    displayed_order : tuple
        Order of only displayed dimensions.
    """

    ndim: int = 2
    ndisplay: Property[int, None, only_2D_3D] = 2
    order: Property[tuple, None, tuple] = ()
    axis_labels: EventedList = field(
        default_factory=EventedList, metadata={'events': False}
    )
    point: EventedList = field(
        default_factory=EventedList, metadata={'events': False}
    )
    range: EventedList = field(
        default_factory=EventedList, metadata={'events': False}
    )
    clip: bool = True

    def __post_init__(self):
        if len(self.axis_labels) > 0:
            self.ndim = len(self.axis_labels)
        elif len(self.order) > 0:
            self.ndim = len(self.order)
        else:
            self._update_lists(self.ndim)

    def _on_order_set(self, order):
        if not set(order) == set(range(self.ndim)):
            raise ValueError(
                f"Invalid ordering {order} for {self.ndim} dimensions"
            )

    def _on_ndim_set(self, ndim):
        self._update_lists(ndim)

    def _update_lists(self, ndim):
        # Point is the slider value
        while len(self.point) < ndim:
            self.point.insert(0, 0)
        while len(self.point) > ndim:
            self.point.pop(0)

        # Range value is (min, max, step) for the entire slider
        while len(self.range) < ndim:
            self.range.insert(0, (0, 2, 1))
        while len(self.range) > ndim:
            self.range.pop(0)

        # Axis labels are strings
        cur_ndim = len(self.axis_labels)
        total = ndim - cur_ndim - 1
        if ndim > cur_ndim:
            for i in range(ndim - cur_ndim):
                total = ndim - cur_ndim - 1
                self.axis_labels.insert(0, str(total - i))
            # If axis labels were previouly ordered, preserve full ordering
            if list(self.axis_labels[ndim - cur_ndim :]) == list(
                map(str, range(cur_ndim))
            ):
                for i in range(ndim - cur_ndim, ndim):
                    self.axis_labels[i] = str(i)
        while len(self.axis_labels) > ndim:
            self.axis_labels.pop(0)

        cur_ndim = len(self.order)
        if ndim > cur_ndim:
            self._order = tuple(range(ndim - cur_ndim)) + tuple(
                [o + ndim - cur_ndim for o in self.order]
            )
        elif cur_ndim < ndim:
            self._order = tuple(
                reorder_after_dim_reduction(self.order[-ndim:])
            )

    @property
    def max_indices(self):
        """Maximum index for each dimension (in data space).
        """
        return [((ma - st) // st) for mi, ma, st in self._range]

    @property
    def indices(self):
        """Tuple of slice objects for slicing arrays on each dimension."""
        slice_list = []
        for axis in range(self.ndim):
            if axis in self.displayed:
                slice_list.append(slice(None))
            else:
                if self.clip:
                    p = np.clip(
                        self.point[axis],
                        np.round(self.range[axis][0]),
                        np.round(self.range[axis][1]) - 1,
                    )
                else:
                    p = self.point[axis]
                p = np.round(p / self.range[axis][2]).astype(int)
                slice_list.append(p)
        return tuple(slice_list)

    @property
    def displayed(self):
        """Tuple: Dimensions that are displayed."""
        return list(self.order[-self.ndisplay :])

    @property
    def not_displayed(self):
        """Tuple: Dimensions that are not displayed."""
        return list(self.order[: -self.ndisplay])

    @property
    def displayed_order(self):
        """Tuple: Order of only displayed dimensions."""
        order = np.array(self.displayed)
        order[np.argsort(order)] = list(range(len(order)))
        return tuple(order)

    def reset(self):
        """Reset dims values to initial states."""
        for axis in range(self.ndim):
            # Range value is (min, max, step) for the entire slider
            self.range[axis] = (0, 2, 1)
            # Point is the slider value
            self.point[axis] = 0
            # Default axis labels go from "-ndim" to "-1" so new axes can easily be added
            self.axis_labels[axis] = str(axis - self.ndim)
        self.order = tuple(range(self.ndim))

    def _roll(self):
        """Roll order of dimensions for display."""
        self.order = np.roll(self.order, 1)

    def _transpose(self):
        """Transpose displayed dimensions."""
        order = list(self.order)
        order[-2], order[-1] = order[-1], order[-2]
        self.order = order

    def set_range(self, axis: int, value: Sequence[Union[int, float]]):
        """Sets the range (min, max, step) for a given dimension.

        Parameters
        ----------
        axis : int
            Dimension index.
        value : tuple
            Range specified as (min, max, step).
        """
        warnings.warn(
            f'To be deprecated, use self.range[axis] =' f' value instead.'
        )
        self.range[axis] = value

    def set_point(self, axis: int, value: Union[int, float]):
        """Sets the point at which to slice this dimension.

        Parameters
        ----------
        axis : int
            Dimension index.
        value : int or float
            Value of the point.
        """
        warnings.warn(
            f'To be deprecated, use self.point[axis] =' f' value instead.'
        )
        self.point[axis] = value

    def set_axis_label(self, axis: int, label: str):
        """Sets a new axis label for the given axis.

        Parameters
        ----------
        axis : int
            Dimension index
        label : str
            Given label
        """
        warnings.warn(
            f'To be deprecated, use self.axis_labels[axis] ='
            f' value instead.'
        )
        self.axis_labels[axis] = label
