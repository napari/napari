from .qt import QtCenter
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QCursor, QPixmap

from ..util.misc import (compute_max_shape as _compute_max_shape,
                         guess_metadata)
from numpy import clip, integer, ndarray, append, insert, delete
from copy import copy

class Center:
    """Center containing the rendered scene, layers, and controlling elements
    including dimension sliders.

    Parameters
    ----------
    parent : Window
        Parent window.

    Attributes
    ----------
    camera : vispy.scene.Camera
        Viewer camera.
    indices : list
        Slicing indices controlled by the sliders.
    max_dims : int
        Maximum dimensions of the contained layers.
    max_shape : tuple of int
        Maximum shape of the contained layers.
    """
    def __init__(self, viewer):

        self.viewer = viewer

        self._qt = QtCenter(self)

        self._qt.canvas.connect(self.on_mouse_move)
        self._qt.canvas.connect(self.on_mouse_press)
        self._qt.canvas.connect(self.on_key_press)
        self._qt.canvas.connect(self.on_key_release)
        # TODO: allow arbitrary display axis setting
        # self.y_axis = 0  # typically the y-axis
        # self.x_axis = 1  # typically the x-axis

        # TODO: wrap indices in custom data structure
        self.indices = [slice(None), slice(None)]

        self._max_dims = 0
        self._max_shape = tuple()

        # update flags
        self._child_layer_changed = False
        self._need_redraw = False
        self._need_slider_update = False

        self._recalc_max_dims = False
        self._recalc_max_shape = False

        self._pos = [0, 0]
        self._index = [0, 0]


    @property
    def _canvas(self):
        return self._qt.canvas

    @property
    def _view(self):
        return self._qt.view

    @property
    def camera(self):
        """vispy.scene.Camera: Viewer camera.
        """
        return self._view.camera

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

    def _axis_to_row(self, axis):
        dims = len(self.indices)
        message = f'axis {axis} out of bounds for {dims} dims'

        if axis < 0:
            axis = dims - axis
            if axis < 0:
                raise IndexError(message)
        elif axis >= dims:
            raise IndexError(message)

        if axis < 2:
            raise ValueError('cannot convert y/x-axes to rows')

        return axis - 1

    def reset_view(self):
        """Resets the camera's view.
        """
        try:
            self.camera.set_range()
        except AttributeError:
            pass

    def screenshot(self, *args, **kwargs):
        """Renders the current canvas.

        Returns
        -------
        screenshot : np.ndarray
            View of the current canvas.
        """
        return self._canvas.render(*args, **kwargs)

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

            self._qt.update_slider(dim, dim_len)

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

    def _update_index(self, event):
        visual = self.viewer.layers[0]._node
        tr = self._canvas.scene.node_transform(self.viewer.layers[0]._node)
        pos = tr.map(event.pos)
        pos = [clip(pos[1],0,self.max_shape[0]-1), clip(pos[0],0,self.max_shape[1]-1)]
        self._index = copy(self.indices)
        self._index[0] = int(pos[0])
        self._index[1] = int(pos[1])

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
            self._update_layers()

        if self._recalc_max_dims:
            self._recalc_max_dims = False
            self._calc_max_dims()

        if self._recalc_max_shape:
            self._recalc_max_shape = False
            self._calc_max_shape()

        if self._need_slider_update:
            self._need_slider_update = False
            self._update_sliders()

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.
        """
        if event.pos is None:
            return

        self._update_index(event)
        if event.is_dragging:
            if self.viewer.annotation and 'Shift' in event.modifiers:
                if self.viewer._active_markers:
                    layer = self.viewer.layers[self.viewer._active_markers]
                    index = layer._selected_markers
                    if index is None:
                        pass
                    else:
                        layer.data[index] = [self._index[1],self._index[0],*self._index[2:]]
                        layer._refresh()
                        self.viewer._update_statusBar()
        else:
            self.viewer._update_active_layers()
            self.viewer._update_statusBar()

    def on_mouse_press(self, event):
        if event.pos is None:
            return
        if self.viewer.annotation:
            if self.viewer._active_markers:
                layer = self.viewer.layers[self.viewer._active_markers]
                if 'Meta' in event.modifiers:
                    index = layer._selected_markers
                    if index is None:
                        pass
                    else:
                        if isinstance(layer.size, (list, ndarray)):
                            layer._size = delete(layer.size, index)
                        layer.data = delete(layer.data, index, axis=0)
                        layer._selected_markers = None
                        self.viewer._update_statusBar()
                elif 'Shift' in event.modifiers:
                    pass
                else:
                    if isinstance(layer.size, (list, ndarray)):
                        layer._size = append(layer.size, 10)
                    coord = [self._index[1],self._index[0],*self._index[2:]]
                    layer.data = append(layer.data, [coord], axis=0)
                    layer._selected_markers = len(layer.data)-1
                    self.viewer._update_statusBar()

    def on_key_press(self, event):
        if event.native.isAutoRepeat():
            return
        else:
            if event.key == ' ':
                if self.viewer.annotation:
                    self.viewer._annotation_history = True
                    self._qt.view.interactive = True
                    self.viewer.annotation = False
                    self._qt.canvas.native.setCursor(QCursor())
                else:
                    self.viewer._annotation_history = False
            elif event.key == 'Shift':
                if self.viewer.annotation and self.viewer._active_markers:
                    self._qt.canvas.native.setCursor(Qt.PointingHandCursor)
            elif event.key == 'Meta':
                if self.viewer.annotation and self.viewer._active_markers:
                    self._qt.canvas.native.setCursor(Qt.ForbiddenCursor)
            elif event.key == 'a':
                cb = self.viewer.layers._qt.layersControls.annotationCheckBox
                cb.setChecked(not cb.isChecked())

    def on_key_release(self, event):
        if event.key == ' ':
            if self.viewer._annotation_history:
                self._qt.view.interactive = False
                self.viewer.annotation = True
                if self.viewer._active_markers:
                    self._qt.canvas.native.setCursor(Qt.CrossCursor)
                else:
                    self._qt.canvas.native.setCursor(self.viewer._disabled_cursor)
        elif event.key == 'Shift':
            if self.viewer.annotation:
                if self.viewer._active_markers:
                    self._qt.canvas.native.setCursor(Qt.CrossCursor)
                else:
                    self._qt.canvas.native.setCursor(self.viewer._disabled_cursor)
        elif event.key == 'Meta':
                if self.viewer._active_markers:
                    self._qt.canvas.native.setCursor(Qt.CrossCursor)
                else:
                    self._qt.canvas.native.setCursor(self.viewer._disabled_cursor)
