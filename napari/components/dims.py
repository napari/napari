from copy import copy
from typing import Union, Sequence
import numpy as np

from .dims_constants import DimsMode
from ..util.event import EmitterGroup


class Dims:
    """Dimensions object modeling multi-dimensional slicing, cropping, and
    displaying in Napari

    Parameters
    ----------
    init_ndim : int, optional
        Initial number of dimensions

    Attributes
    ----------
    events : EmitterGroup
        Event emitter group
    range : list of 3-tuple
        List of tuples (min, max, step), one for each dimension
    point : list of float
        List of floats setting the current value of the range slider when in
        POINT mode, one for each dimension
    interval : list of 2-tuple
        List of tuples (min, max) setting the current selection of the range
        slider when in INTERVAL mode, one for each dimension
    mode : list of DimsMode
        List of DimsMode, one for each dimension
    clip : bool
        Flag if to clip indices based on range. Needed for image-like
        layers, but prevents shape-like layers from adding new shapes
        outside their range.
    order : tuple of int
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    ndim : int
        Number of dimensions.
    ndisplay : int
        Number of displayed dimensions.
    indices : tuple of slice object
        Tuple of slice objects for slicing arrays on each dimension, one for
        each dimension
    displayed : tuple
        List of dimensions that are displayed.
    not_displayed : tuple
        List of dimensions that are not displayed.
    displayed_order : tuple
        Order of only displayed dimensions.
    """

    def __init__(self, init_ndim=0):
        super().__init__()

        # Events:
        self.events = EmitterGroup(
            source=self,
            auto_connect=True,
            axis=None,
            ndim=None,
            ndisplay=None,
            order=None,
            range=None,
            camera=None,
        )

        self._range = []
        self._point = []
        self._interval = []
        self._mode = []
        self._order = []
        self.clip = True

        self._ndisplay = 2
        self.ndim = init_ndim

    def __str__(self):
        return "~~".join(
            map(
                str,
                [self.range, self.point, self.interval, self.mode, self.order],
            )
        )

    @property
    def range(self):
        """List of 3-tuple (min, max, step): total range and step size of each
        dimension.
        """
        return copy(self._range)

    @property
    def point(self):
        """List of int: value of each dimension if in POINT mode.
        """
        return copy(self._point)

    @property
    def interval(self):
        """List of 2-tuple (min, max): Selection range of each dimension if in
        INTERVAL mode.
        """
        return copy(self._interval)

    @property
    def mode(self):
        """List of DimsMode: List of DimsMode, one for each dimension."""
        return copy(self._mode)

    @property
    def order(self):
        """List of int: Display order of dimensions."""
        return self._order

    @order.setter
    def order(self, order):
        if np.all(self._order == order):
            return

        if not len(order) == self.ndim:
            raise ValueError(
                f"Invalid ordering {order} for {self.ndim} dimensions"
            )
        if np.all(self._order == order):
            return
        self._order = order
        self.events.order()
        self.events.camera()

    @property
    def ndim(self):
        """Returns the number of dimensions

        Returns
        -------
        ndim : int
            Number of dimensions
        """
        return len(self.point)

    @ndim.setter
    def ndim(self, ndim):
        if self.ndim == ndim:
            return
        elif self.ndim < ndim:
            for i in range(self.ndim, ndim):
                self.set_initial_dims(0, insert=True)
            # Notify listeners that the number of dimensions have changed
            self.events.ndim()

            # Notify listeners of which dimensions have been affected
            for axis_changed in range(ndim - self.ndim):
                self.events.axis(axis=axis_changed)
        elif self.ndim > ndim:
            self._range = self._range[-ndim:]
            self._point = self._point[-ndim:]
            self._interval = self._interval[-ndim:]
            self._mode = self._mode[-ndim:]
            order = np.array(self._order[-ndim:])
            order[np.argsort(order)] = list(range(len(order)))
            self._order = list(order)

            # Notify listeners that the number of dimensions have changed
            self.events.ndim()

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
    def ndisplay(self):
        """Int: Number of displayed dimensions."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay):
        if self._ndisplay == ndisplay:
            return

        if ndisplay not in (2, 3):
            raise ValueError(
                f"Invalid number of dimensions to be displayed {ndisplay}"
            )

        self._ndisplay = ndisplay
        self.events.ndisplay()
        self.events.camera()

    @property
    def displayed(self):
        """Tuple: Dimensions that are displayed."""
        return self.order[-self.ndisplay :]

    @property
    def not_displayed(self):
        """Tuple: Dimensions that are not displayed."""
        return self.order[: -self.ndisplay]

    @property
    def displayed_order(self):
        """Tuple: Order of only displayed dimensions."""
        order = np.array(self.displayed)
        order[np.argsort(order)] = list(range(len(order)))
        return tuple(order)

    def set_initial_dims(self, axis, insert=False):
        """Initializes the dimensions values for a given axis (dimension)

        Parameters
        ----------
        axis : int
            Dimension index
        insert : bool
            Whether to insert the axis or not during initialization
        """
        if insert:
            # Insert default values
            # Range value is (min, max, step) for the entire slider
            self._range.insert(axis, (0, 2, 1))
            # Point is the slider value if in point mode
            self._point.insert(axis, 0)
            # Interval value is the (min, max) of the slider selction
            # if in interval mode
            self._interval.insert(axis, (0, 1))
            self._mode.insert(axis, DimsMode.POINT)
            cur_order = [o if o < axis else o + 1 for o in self.order]
            self._order = [axis] + cur_order
        else:
            # Range value is (min, max, step) for the entire slider
            self._range[axis] = (0, 2, 1)
            # Point is the slider value if in point mode
            self._point[axis] = 0
            # Interval value is the (min, max) of the slider selction
            # if in interval mode
            self._interval[axis] = (0, 1)
            self._mode[axis] = DimsMode.POINT
            self._order[axis] = axis

    def set_range(self, axis: int, range: Sequence[Union[int, float]]):
        """Sets the range (min, max, step) for a given axis (dimension)

        Parameters
        ----------
        axis : int
            Dimension index
        range : tuple
            Range specified as (min, max, step)
        """
        if axis < 0:
            axis += self.ndim
        if axis < 0:
            raise ValueError(
                f'axis is negative, expected positive, got {axis}'
            )
        if self.range[axis] != range:
            self._range[axis] = range
            self.events.range(axis=axis)

    def set_point(self, axis: int, value: Union[int, float]):
        """Sets the point at which to slice this dimension

        Parameters
        ----------
        axis : int
            Dimension index
        value : int or float
            Value of the point
        """
        if axis < 0:
            axis += self.ndim
        if axis < 0:
            raise ValueError(
                f'axis is negative, expected positive, got {axis}'
            )
        if self.point[axis] != value:
            self._point[axis] = value
            self.events.axis(axis=axis)

    def set_interval(self, axis: int, interval: Sequence[Union[int, float]]):
        """Sets the interval used for cropping and projecting this dimension

        Parameters
        ----------
        axis : int
            Dimension index
        interval : tuple
            INTERVAL specified with (min, max)
        """
        if axis < 0:
            axis += self.ndim
        if axis < 0:
            raise ValueError(
                f'axis is negative, expected positive, got {axis}'
            )
        if self.interval[axis] != interval:
            self._interval[axis] = interval
            self.events.axis(axis=axis)

    def set_mode(self, axis: int, mode: DimsMode):
        """Sets the mode: POINT or INTERVAL

        Parameters
        ----------
        axis : int
            Dimension index
        mode : POINT or INTERVAL
            Whether dimension is in the POINT or INTERVAL mode
        """
        if axis < 0:
            axis += self.ndim
        if axis < 0:
            raise ValueError(
                f'axis is negative, expected positive, got {axis}'
            )
        if self.mode[axis] != mode:
            self._mode[axis] = mode
            self.events.axis(axis=axis)

    def _roll(self):
        """Roll order of dimensions for display."""
        self.order = np.roll(self.order, 1)

    def _transpose(self):
        """Transpose displayed dimensions."""
        order = copy(self.order)
        order[-2], order[-1] = order[-1], order[-2]
        self.order = order
