import numpy as np
from copy import copy
from enum import Enum
from typing import Union, Tuple, Iterable, Sequence

from ...util.event import EmitterGroup


class DimsMode(Enum):
    POINT = 0
    INTERVAL = 1


class Dims():
    """Dimensions object modeling multi-dimensional slicing, cropping, and
    displaying in Napari

    Parameters
    ----------
    init_ndims : int, optional
        Initial number of dimensions

    Attributes
    ----------
    events : EmitterGroup
        Event emitter group
    range : list
        List of tuples (min, max, step), one for each dimension
    point : list
        List of floats, one for each dimension
    interval : list
        List of tuples (min, max), one for each dimension
    mode : list
        List of DimsMode, one for each dimension
    display : list
        List of bool indicating if dimension displayed or not, one for each
        dimension
    ndims : int
        Number of dimensions
    displayed : np.ndarray
        Array of the displayed dimensions
    """
    def __init__(self, init_ndims=0):
        super().__init__()

        # Events:
        self.events = EmitterGroup(source=self, auto_connect=True, axis=None,
                                   ndims=None)

        self.range = []
        self.point = []
        self.interval = []
        self.mode = []
        self.display = []

        self._add_axes(init_ndims - 1)

    def __str__(self):
        return "~~".join([str(self.range),
                         str(self.point),
                         str(self.interval),
                         str(self.mode),
                         str(self.display)])

    @property
    def ndims(self):
        """Returns the number of dimensions

        Returns
        -------
        ndims : int
            Number of dimensions
        """
        return len(self.point)

    @ndims.setter
    def ndims(self, ndims):
        if self.ndims < ndims:
            self._add_axes(ndims - 1)
        elif self.ndims > ndims:
            self._trim_ndims(ndims)

    @property
    def displayed(self):
        """Returns the displayed dimensions

        Returns
        -------
        displayed : np.ndarray
            Displayed dimensions
        """
        displayed_one_hot = copy(self.display)
        displayed_one_hot = ([False if ind is None else ind for ind in
                             displayed_one_hot])
        return np.nonzero(list(displayed_one_hot))[0]

    @property
    def slice_and_project(self):
        """Returns the slice and project tuples that specify how to slice and
        project arrays.

        Returns
        -------
        slice : tuple
            The slice tuple
        project : tuple
            The projection tuple
        """

        slice_list = []
        project_list = []
        z = zip(self.mode, self.display, self.point, self.interval, self.range)
        for (mode, display, point, interval, range) in z:
            if mode == DimsMode.POINT or mode is None:
                if display:
                    # no slicing, cropping or projection:
                    project_list.append(False)
                    slice_list.append(slice(None, None, None))
                else:
                    # slice:
                    project_list.append(False)
                    slice_list.append(int(round(point)))
            elif mode == DimsMode.INTERVAL:
                if display:
                    # crop for display:
                    project_list.append(False)
                    if interval is None:
                        slice_list.append(slice(None))
                    else:
                        slice_list.append(slice(int(round(interval[0])),
                                          int(round(interval[1]))))
                else:
                    # crop before project:
                    project_list.append(True)
                    if interval is None:
                        slice_list.append(slice(None))
                    else:
                        slice_list.append(slice(int(round(interval[0])),
                                          int(round(interval[1]))))

        slice_tuple = tuple(slice_list)
        project_tuple = tuple(project_list)

        return slice_tuple, project_tuple

    @property
    def indices(self):
        """
        Indices for slicing

        Returns
        -------
        slice : tuple
            The slice tuple
        """
        return self.slice_and_project[0]

    def set_all_ranges(self, all_ranges: Sequence[Union[int, float]]):
        """Sets ranges for all dimensions

        Parameters
        ----------
        ranges : tuple
            Ranges of all dimensions
        """
        ndim = len(all_ranges)
        modified_dims = self._add_axes(ndim-1, no_event=True)
        self._set_2d_viewing()
        self.range = all_ranges

        self.events.ndims()
        for axis_changed in modified_dims:
            self.events.axis(axis=axis_changed)

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
            self.range[axis] = range
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
            self.point[axis] = value
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
            self.interval[axis] = interval
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
            self.mode[axis] = mode
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
            self.display[axis] = display
            self.events.axis(axis=axis)

    def _set_2d_viewing(self):
        """Sets the 2d viewing
        """
        self.display = [False] * len(self.display)
        self.display[-1] = True
        self.display[-2] = True

    def _add_axes(self, axis: int, no_event=None):
        """Makes sure that the given axis is in the dimension model

        Parameters
        ----------
        axis : int
            Dimension index

        Returns
        -------
        dimensions : list
            List of axes
        """
        if axis >= self.ndims:
            old_ndims = self.ndims
            margin_length = 1 + axis - self.ndims
            self.range.extend([(0.0, 1.0, 0.01)] * (margin_length))
            self.point.extend([0.0] * (margin_length))
            self.interval.extend([(0.3, 0.7)] * (margin_length))
            self.mode.extend([DimsMode.POINT] * (margin_length))
            self.display.extend([False] * (margin_length))

            if not no_event:
                # Notify listeners that the number of dimensions have changed
                self.events.ndims()

                # Notify listeners of which dimensions have been affected
                for axis_changed in range(old_ndims - 1, self.ndims):
                    self.events.axis(axis=axis_changed)

            return list(range(old_ndims, 1 + axis))

        return []

    def _trim_ndims(self, ndims: int):
        """This internal method is used to trim the number of axis.

        Parameters
        ----------
        ndims : int
            The new number of dimensions
        """
        if ndims < self.ndims:
            self.range = self.range[:ndims]
            self.point = self.point[:ndims]
            self.interval = self.interval[:ndims]
            self.mode = self.mode[:ndims]
            self.display = self.display[:ndims]

            # Notify listeners that the number of dimensions have changed
            self.events.ndims()
