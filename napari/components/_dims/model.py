from ...util.misc import (compute_max_shape as _compute_max_shape,
                          guess_metadata)
from numpy import clip, integer, ndarray, append, insert, delete
from copy import copy

from vispy.util.event import EmitterGroup, Event

#from .view import QtDims

from enum import Enum


class Mode(Enum):
     Display = 0
     Slice = 1
     Project = 2


class Dims:
    """Dimensions object containing the dimension sliders

    Attributes
    ----------
    indices : list
        Slicing indices controlled by the sliders.
    max_dims : int
        Maximum dimensions of the contained layers.
    max_shape : tuple of int
        Maximum shape of the contained layers.
    """
    def __init__(self, max_dims):

        self.events = EmitterGroup(source=self,
                                   auto_connect=True,
                                   update_slider=Event)


        self.viewer = None

        self._max_dims = max_dims
        self._max_shape = tuple()

        self.point = []
        self.mode  = []

        self.set_point(max_dims-1,0)


        # update flags
        self._child_layer_changed = False
        self._need_redraw = False
        self._need_slider_update = False

        self._recalc_max_dims = False
        self._recalc_max_shape = False

        self._qt = None #QtDims(self)




    def set_point(self, axis, value):
        self._ensure_correct_lengths(axis)
        self.point[axis] = value
        self._need_redraw = True
        self._update()

    def set_mode(self, axis, mode):
        self._ensure_correct_lengths(axis)
        self.mode[axis]=mode
        self._need_redraw = True
        self._update()

    def _ensure_correct_lengths(self, axis):
        if axis >= len(self.mode):
            self.point.extend([0.0]*(1+ axis - len(self.mode)))
            self.mode.extend([Mode.Slice] * (1 + axis - len(self.mode)))

    @property
    def arrayslice(self):
        return


    @property
    def max_dims(self):
        """int: Maximum tunable dimensions for contained images.
        """
        return self._max_dims

    @property
    def max_shape(self):
        """tuple: Maximum shape for contained images.
        """
        return self._max_shape


    def _slider_value_changed(self, value, axis):
        self.set_point(axis, value)

    def _update_sliders(self):
        """Updates the sliders according to the contained images.
        """
        max_dims = self.max_dims
        max_shape = self.max_shape

        curr_dims = len(self.indices)

        if curr_dims > max_dims:
            self.indices = self.indices[:max_dims]
            dims = curr_dims
        else:
            dims = max_dims
            self.indices.extend([0] * (max_dims - curr_dims))

        for dim in range(2, dims):  # do not create sliders for y/x-axes
            try:
                dim_len = max_shape[dim]
            except IndexError:
                dim_len = 0
            self.events.update_slider(dim=dim,
                                      dim_len=dim_len,
                                      max_dims=max_dims)



    def _calc_max_dims(self):
        """Calculates the number of maximum dimensions in the contained images.
        """
        max_dims = 0

        for layer in self.viewer.layers:
            dims = layer.ndim
            if dims > max_dims:
                max_dims = dims

        self._max_dims = max_dims

        self._need_slider_update = True
        self._update()

    def _calc_max_shape(self):
        """Calculates the maximum shape of the contained images.
        """
        shapes = (layer.shape for layer in self.viewer.layers)
        self._max_shape = _compute_max_shape(shapes, self.max_dims)

    def _on_layers_change(self, event):
        """Called whenever a layer is changed.
        """
        self._child_layer_changed = True
        self._update()

    def _update(self):
        """Updates the viewer.
        """
        if self._child_layer_changed:
            self._child_layer_changed = False
            self._recalc_max_dims = True
            self._recalc_max_shape = True
            self._need_slider_update = True

        if self._need_redraw:
            self._need_redraw = False
            self.viewer._update_layers()

        if self._recalc_max_dims:
            self._recalc_max_dims = False
            self._calc_max_dims()

        if self._recalc_max_shape:
            self._recalc_max_shape = False
            self._calc_max_shape()

        if self._need_slider_update:
            self._need_slider_update = False
            self._update_sliders()
