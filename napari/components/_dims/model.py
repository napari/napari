from math import inf

from typing import Union, Tuple

from napari.components.component import Component
from napari.util.event import EmitterGroup
from ...util.misc import (compute_max_shape as _compute_max_shape,
                          guess_metadata)
from enum import Enum


class DimsMode(Enum):
      Point = 0
      Interval = 1



class Dims(Component) :


    def __init__(self, init_max_dims=0):
        """
        Dimensions object modelling multi-dimensional slicing, cropping, and displaying in Napari
        Parameters
        ----------
        init_max_dims : initial number of dimensions
        """
        super().__init__()

        # Events:
        self.changed = EmitterGroup(source=self,
                                    auto_connect=True,
                                    axis=None,
                                    nbdims=None)


        self.range    = []
        self.point    = []
        self.interval = []
        self.mode     = []
        self.display  = []

        self._ensure_axis_present(init_max_dims-1)


    @property
    def num_dimensions(self):
        """
        Returns the number of dimensions
        Returns numebr of dimensions
        -------

        """
        return len(self.point)

    @num_dimensions.setter
    def num_dimensions(self, num_dimensions):
        if self.num_dimensions < num_dimensions:
            self._ensure_axis_present(num_dimensions - 1)
        elif self.num_dimensions > num_dimensions:
            self._trim_nb_dimensions(num_dimensions)


    def set_range(self, axis: int, range: Tuple[Union[int,float]]):
        """
        Sets the range (min, max, step) for a given axis (dimension)
        Parameters
        ----------
        axis : dimension index
        range : (min, max, step) tuple
        """
        self._ensure_axis_present(axis)
        if self.range[axis] != range:
            self.range[axis] = range
            self.changed.axis(axis=axis)

    def get_range(self, axis):
        """
        Returns the point at which this dimension is sliced
        Parameters
        ----------
        axis : dimension index
        """
        return self.range[axis]

    def set_point(self, axis: int, value: Union[int,float]):
        """
        Sets thepoint at which to slice this dimension
        Parameters
        ----------
        axis : dimension index
        value :
        """
        self._ensure_axis_present(axis)
        if self.point[axis] != value:
            self.point[axis] = value
            self.changed.axis(axis=axis)

    def get_point(self, axis):
        """
        Returns the point at which this dimension is sliced
        Parameters
        ----------
        axis : dimension index
        """
        return self.point[axis]

    def set_interval(self, axis: int, interval: Tuple[Union[int, float]]):
        """
        Sets the interval used for cropping and projecting this dimension
        Parameters
        ----------
        axis : dimension index
        interval : (min, max) tuple
        """
        self._ensure_axis_present(axis)
        if self.interval[axis] != interval:
            self.interval[axis] = interval
            self.changed.axis(axis=axis)

    def get_interval(self, axis):
        """

        Parameters
        ----------
        axis : dimension index
        """
        return self.interval[axis]

    def set_mode(self, axis: int, mode:DimsMode):
        """
        Sets the mode: Point or Interval
        Parameters
        ----------
        axis : dimension index
        mode : Point or Interval
        """
        self._ensure_axis_present(axis)
        if self.mode[axis]!=mode:
            self.mode[axis] = mode
            self.changed.axis(axis=axis)

    def get_mode(self, axis):
        """
        Returns the mode for a given axis
        Parameters
        ----------
        axis : dimension index
        """
        return self.mode[axis]

    def set_display(self, axis: int, display:bool):
        """
        Sets the display boolean flag for a given axis
        Parameters
        ----------
        axis : dimension index
        display : True for display, False for slice or project...
        """
        self._ensure_axis_present(axis)
        if self.display[axis]!=display:
            self.display[axis] = display
            self.changed.axis(axis=axis)

    def get_display(self, axis):
        """
        retruns the display boolean flag for a given axis
        Parameters
        ----------
        axis : dimension index
        """
        return self.display[axis]


    def _ensure_axis_present(self, axis: int):
        """
        Makes sure that the given axis is in the dimension model
        Parameters.
        ----------
        axis : axis index
        """
        if axis >= self.num_dimensions:
            old_nb_dimensions = self.num_dimensions
            margin_length = 1+ axis - self.num_dimensions
            self.range.extend([(None,None,None)] * (margin_length))
            self.point.extend([0.0]*(margin_length))
            self.interval.extend([None] * (margin_length))
            self.mode.extend([None] * (margin_length))
            self.display.extend([False] * (margin_length))

            # First we notify listeners that the number of dimensions have changed:
            self.changed.nbdims()

            # Then we notify listeners of which dimensions have been affected.
            for axis_changed in range(old_nb_dimensions-1, self.num_dimensions):
                self.changed.axis(axis=axis_changed)


    def _trim_nb_dimensions(self, nb_dimensions):
        """
        This internal method is used to trim the number of axis.
        Parameters
        ----------
        nb_dimensions : new number of dimensions, must be less that
        """
        if nb_dimensions<self.num_dimensions:
            self.range = self.range[:nb_dimensions]
            self.point = self.point[:nb_dimensions]
            self.interval = self.interval[:nb_dimensions]
            self.mode = self.mode[:nb_dimensions]
            self.display = self.display[:nb_dimensions]

            # First we notify listeners that the number of dimensions have changed:
            self.changed.nbdims()


    @property
    def slice_and_project(self):
        """
        Returns the slice and project tuples that specify how to slice and project arrays.
        Returns (slice, project)
        -------

        Parameters
        ----------

        """

        slice_list = []
        project_list = []

        for  (mode, display, point, interval) in zip(self.mode, self.display, self.point, self.interval):

            if mode   == DimsMode.Point or mode is None:
                if display:
                    # no slicing, cropping or projection:
                    project_list.append(False)
                    slice_list.append(slice(None))
                else:
                    # slice:
                    project_list.append(False)
                    slice_list.append(slice(round(point)))
            elif mode == DimsMode.Interval:
                if display:
                    # crop for display:
                    project_list.append(False)
                    if interval is None :
                        slice_list.append(slice(None))
                    else:
                        slice_list.append(slice(*interval))
                else:
                    # crop before project:
                    project_list.append(True)
                    if interval is None:
                        slice_list.append(slice(None))
                    else:
                        slice_list.append(slice(*interval))

        slice_tuple = tuple(slice_list)
        project_tuple = tuple(project_list)

        return slice_tuple, project_tuple

