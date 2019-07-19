from copy import copy
from typing import Union, Sequence

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
    display : list of bool
        List of bool indicating if dimension displayed or not, one for each
        dimension
    ndim : int
        Number of dimensions
    indices : tuple of slice object
        Tuple of slice objects for slicing arrays on each dimension, one for
        each dimension
    """

    def __init__(self, init_ndim=0):
        super().__init__()

        # Events:
        self.events = EmitterGroup(
            source=self, auto_connect=True, axis=None, ndim=None
        )

        self._range = []
        self._point = []
        self._interval = []
        self._mode = []
        self._display = []

        self.ndim = init_ndim

    def __str__(self):
        return "~~".join(
            map(
                str,
                [
                    self.range,
                    self.point,
                    self.interval,
                    self.mode,
                    self.display,
                ],
            )
        )

    @property
    def range(self):
        """list of 3-tuple (min, max, step): total range and step size of each
        dimension.
        """
        return copy(self._range)

    @range.setter
    def range(self, value):
        if value == self.range:
            return
        self._range = value
        self.ndim = len(value)
        for axis in range(self.ndim):
            self.events.axis(axis=axis)

    @property
    def point(self):
        """list of int: value of each dimension if in POINT mode.
        """
        return copy(self._point)

    @property
    def interval(self):
        """list of 2-tuple (min, max): Selection range of each dimension if in
        INTERVAL mode.
        """
        return copy(self._interval)

    @property
    def mode(self):
        """list of DimsMode: List of DimsMode, one for each dimension
        """
        return copy(self._mode)

    @property
    def display(self):
        """list: List of bool indicating if dimension displayed or not, one for
        each dimension
        """
        return copy(self._display)

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
        if ndim > self.ndim:
            for i in range(self.ndim, ndim):
                # Insert default values for the new axis at the beginning of
                # the lists
                # Range value is (min, max, step) for the entire slider
                self._range.insert(0, (0, 2, 1))
                # Point is the slider value if in point mode
                self._point.insert(0, 0)
                # Interval value is the (min, max) of the slider selction
                # if in interval mode
                self._interval.insert(0, (0, 1))
                self._mode.insert(0, DimsMode.POINT)
                self._display.insert(0, False)

            # Notify listeners that the number of dimensions have changed
            self.events.ndim()

            # Notify listeners of which dimensions have been affected
            for axis_changed in range(ndim - self.ndim):
                self.events.axis(axis=axis_changed)

        elif ndim < self.ndim:
            self._range = self._range[-ndim:]
            self._point = self._point[-ndim:]
            self._interval = self._interval[-ndim:]
            self._mode = self._mode[-ndim:]
            self._display = self._display[-ndim:]

            # Notify listeners that the number of dimensions have changed
            self.events.ndim()

    @property
    def indices(self):
        """Tuple of slice objects for slicing arrays on each dimension."""
        slice_list = []
        z = zip(self.mode, self.display, self.point, self.interval)
        for (mode, display, point, interval) in z:
            if mode == DimsMode.POINT:
                if display:
                    slice_list.append(slice(None, None, None))
                else:
                    slice_list.append(int(round(point)))
            elif mode == DimsMode.INTERVAL:
                if display:
                    slice_list.append(slice(None, None, None))

                else:
                    slice_list.append(
                        slice(int(round(interval[0])), int(round(interval[1])))
                    )

        return tuple(slice_list)

    def set_range(self, axis: int, range: Sequence[Union[int, float]]):
        """Sets the range (min, max, step) for a given axis (dimension)

        Parameters
        ----------
        axis : int
            Dimension index
        range : tuple
            Range specified as (min, max, step)
        """
        if self.range[axis] != range:
            self._range[axis] = range
            self.events.axis(axis=axis)

    def set_point(self, axis: int, value: Union[int, float]):
        """Sets the point at which to slice this dimension

        Parameters
        ----------
        axis : int
            Dimension index
        value : int or float
            Value of the point
        """
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
        if self.mode[axis] != mode:
            self._mode[axis] = mode
            self.events.axis(axis=axis)

    def set_display(self, axis: int, display: bool):
        """Sets the display boolean flag for a given axis

        Parameters
        ----------
        axis : int
            Dimension index
        display : bool
            Bool which is `True` for display and `False` for slice or project.
        """
        if self.display[axis] != display:
            self._display[axis] = display
            self.events.axis(axis=axis)

    def _set_2d_viewing(self):
        """Sets the 2d viewing
        """
        for i in range(len(self.display) - 2):
            self.set_display(i, False)
        if len(self.display) >= 2:
            self.set_display(-1, True)
            self.set_display(-2, True)
