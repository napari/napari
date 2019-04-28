import numpy as np
from copy import copy
from typing import Union, Tuple, Iterable, Sequence

from ._constants import DimsMode
from ...util.event import EmitterGroup


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
    ndim : int
        Number of dimensions
    displayed : np.ndarray
        Array of the displayed dimensions
    """
    def __init__(self, init_ndim=0):
        super().__init__()

        # Events:
        self.events = EmitterGroup(source=self, auto_connect=True, axis=None,
                                   ndim=None)

        self.range = []
        self.point = []
        self.interval = []
        self.mode = []
        self.display = []

        self.ndim = init_ndim

    def __str__(self):
        return "~~".join(map(str, [self.range, self.point, self.interval,
                                   self.mode, self.display]))

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
                self.range.insert(0, (0.0, 1.0, 0.01))
                self.point.insert(0, 0.0)
                self.interval.insert(0, (0.3, 0.7))
                self.mode.insert(0, DimsMode.POINT)
                self.display.insert(0, False)

            # Notify listeners that the number of dimensions have changed
            self.events.ndim()

            # Notify listeners of which dimensions have been affected
            for axis_changed in range(ndim - self.ndim):
                self.events.axis(axis=axis_changed)

        elif ndim < self.ndim:
            self.range = self.range[-ndim:]
            self.point = self.point[-ndim:]
            self.interval = self.interval[-ndim:]
            self.mode = self.mode[-ndim:]
            self.display = self.display[-ndim:]

            # Notify listeners that the number of dimensions have changed
            self.events.ndim()

    @property
    def displayed(self):
        """Returns the displayed dimensions

        Returns
        -------
        displayed : np.ndarray
            Displayed dimensions
        """
        displayed_one_hot = copy(self.display)
        displayed_one_hot = [False if ind is None else ind for ind in
                             displayed_one_hot]
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
        self.ndim = len(all_ranges)
        self._set_2d_viewing()
        self.range = all_ranges

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
