from math import inf

from typing import Union, Tuple

from napari.components.component import Component
from ...util.misc import (compute_max_shape as _compute_max_shape,
                          guess_metadata)
from enum import Enum


class Mode(Enum):
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


        self.viewer = None

        self.range    = []
        self.point    = []
        self.interval = []
        self.mode     = []
        self.display  = []

        self._ensure_axis_present(init_max_dims-1)


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
            self._notify_listeners(source=self, axis=axis)

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
            self._notify_listeners(source=self, axis=axis)

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
            self._notify_listeners(source=self, axis=axis)

    def get_interval(self, axis):
        """

        Parameters
        ----------
        axis : dimension index
        """
        return self.interval[axis]

    def set_mode(self, axis: int, mode:Mode):
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
            self._notify_listeners(source=self, axis=axis)

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
            self._notify_listeners(source=self, axis=axis)

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
        Parameters
        ----------
        axis : axis index
        """
        if axis >= self.nb_dimensions:
            margin_length = 1+ axis - self.nb_dimensions
            self.range.extend([(None,None,None)] * (margin_length))
            self.point.extend([0.0]*(margin_length))
            self.interval.extend([None] * (margin_length))
            self.mode.extend([None] * (margin_length))
            self.display.extend([False] * (margin_length))

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

            if mode   == Mode.Point or mode is None:
                if display:
                    # no slicing, cropping or projection:
                    project_list.append(False)
                    slice_list.append(slice(None))
                else:
                    # slice:
                    project_list.append(False)
                    slice_list.append(slice(round(point)))
            elif mode == Mode.Interval:
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



    @property
    def nb_dimensions(self):
        """
        Returns the number of dimensions
        Returns numebr of dimensions
        -------

        """
        return len(self.point)



# class Dims:
#     """Dimensions object containing the dimension sliders
#
#     Parameters
#     ----------
#     viewer : Viewer
#         Parent viewer.
#
#     Attributes
#     ----------
#     indices : list
#         Slicing indices controlled by the sliders.
#     max_dims : int
#         Maximum dimensions of the contained layers.
#     max_shape : tuple of int
#         Maximum shape of the contained layers.
#     """
#
#     def __init__(self, viewer):
#
#         self.viewer = viewer
#         self.events = EmitterGroup(source=self,
#                                    auto_connect=True,
#                                    update_slider=Event)
#         # TODO: allow arbitrary display axis setting
#         # self.y_axis = 0  # typically the y-axis
#         # self.x_axis = 1  # typically the x-axis
#
#         # TODO: wrap indices in custom data structure
#         self.indices = [slice(None), slice(None)]
#
#         self._max_dims = 0
#         self._max_shape = tuple()
#
#         # update flags
#         self._child_layer_changed = False
#         self._need_redraw = False
#         self._need_slider_update = False
#
#         self._recalc_max_dims = False
#         self._recalc_max_shape = False
#
#         self._qt = QtDims(self)
#
#     @property
#     def max_dims(self):
#         """int: Maximum tunable dimensions for contained images.
#         """
#         return self._max_dims
#
#     @property
#     def max_shape(self):
#         """tuple: Maximum shape for contained images.
#         """
#         return self._max_shape
#
#     def _update_sliders(self):
#         """Updates the sliders according to the contained images.
#         """
#         max_dims = self.max_dims
#         max_shape = self.max_shape
#
#         curr_dims = len(self.indices)
#
#         if curr_dims > max_dims:
#             self.indices = self.indices[:max_dims]
#             dims = curr_dims
#         else:
#             dims = max_dims
#             self.indices.extend([0] * (max_dims - curr_dims))
#
#         for dim in range(2, dims):  # do not create sliders for y/x-axes
#             try:
#                 dim_len = max_shape[dim]
#             except IndexError:
#                 dim_len = 0
#             self.events.update_slider(dim=dim, dim_len=dim_len,
#                                       max_dims=max_dims)
#
#     def _slider_value_changed(self, value, axis):
#         self.indices[axis] = value
#         self._need_redraw = True
#         self._update()
#
#     def _calc_max_dims(self):
#         """Calculates the number of maximum dimensions in the contained images.
#         """
#         max_dims = 0
#
#         for layer in self.viewer.layers:
#             dims = layer.ndim
#             if dims > max_dims:
#                 max_dims = dims
#
#         self._max_dims = max_dims
#
#         self._need_slider_update = True
#         self._update()
#
#     def _calc_max_shape(self):
#         """Calculates the maximum shape of the contained images.
#         """
#         shapes = (layer.shape for layer in self.viewer.layers)
#         self._max_shape = _compute_max_shape(shapes, self.max_dims)
#
#     def _on_layers_change(self, event):
#         """Called whenever a layer is changed.
#         """
#         self._child_layer_changed = True
#         self._update()
#
#     def _update(self):
#         """Updates the viewer.
#         """
#         if self._child_layer_changed:
#             self._child_layer_changed = False
#             self._recalc_max_dims = True
#             self._recalc_max_shape = True
#             self._need_slider_update = True
#
#         if self._need_redraw:
#             self._need_redraw = False
#             self.viewer._update_layers()
#
#         if self._recalc_max_dims:
#             self._recalc_max_dims = False
#             self._calc_max_dims()
#
#         if self._recalc_max_shape:
#             self._recalc_max_shape = False
#             self._calc_max_shape()
#
#         if self._need_slider_update:
#             self._need_slider_update = False
#             self._update_sliders()

