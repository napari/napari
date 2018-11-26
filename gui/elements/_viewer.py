from .qt import QtViewer

from ..util.misc import (compute_max_shape as _compute_max_shape,
                         guess_metadata)
from numpy import clip, integer, ndarray, append, insert, delete
from copy import copy

class Viewer:
    """Viewer containing the rendered scene, layers, and controlling elements
    including dimension sliders.

    Parameters
    ----------
    parent : Window
        Parent window.

    Attributes
    ----------
    camera : vispy.scene.Camera
        Viewer camera.
    layers : LayerList
        List of contained layers.
    indices : list
        Slicing indices controlled by the sliders.
    window : Window
        Parent window
    max_dims : int
        Maximum dimensions of the contained layers.
    max_shape : tuple of int
        Maximum shape of the contained layers.
    """
    def __init__(self, window):
        from ._layer_list import LayerList
        from ._controls import Controls

        self._window = window

        self._qt = QtViewer(self)
        self._qt.canvas.connect(self.on_mouse_move)
        self._qt.canvas.connect(self.on_mouse_press)
        self._qt.canvas.connect(self.on_mouse_release)
        # TODO: allow arbitrary display axis setting
        # self.y_axis = 0  # typically the y-axis
        # self.x_axis = 1  # typically the x-axis

        # TODO: wrap indices in custom data structure
        self.indices = [slice(None), slice(None)]

        self.layers = LayerList(self)

        self.controls = Controls()

        self._max_dims = 0
        self._max_shape = tuple()

        # update flags
        self._child_layer_changed = False
        self._need_redraw = False
        self._need_slider_update = False

        self._recalc_max_dims = False
        self._recalc_max_shape = False

        self._pos = [0, 0]
        self.interactive = False

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
    def window(self):
        """Window: Parent window.
        """
        return self._window

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

    def add_layer(self, layer):
        """Adds a layer to the viewer.

        Parameters
        ----------
        layer : Layer
            Layer to add.
        """
        self.layers.append(layer)
        if len(self.layers) == 1:
            self.reset_view()

    def imshow(self, image, meta=None, multichannel=None, **kwargs):
        """Shows an image in the viewer.

        Parameters
        ----------
        image : np.ndarray
            Image data.
        meta : dict, optional
            Image metadata.
        multichannel : bool, optional
            Whether the image is multichannel. Guesses if None.
        **kwargs : dict
            Parameters that will be translated to metadata.

        Returns
        -------
        layer : Image
            Layer for the image.
        """
        meta = guess_metadata(image, meta, multichannel, kwargs)

        return self.add_image(image, meta)

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

    def _update_layers(self):
        """Updates the contained layers.
        """
        for layer in self.layers:
            layer._set_view_slice(self.indices)

        self._update_statusBar()

    def _calc_max_dims(self):
        """Calculates the number of maximum dimensions in the contained images.
        """
        max_dims = 0

        for layer in self.layers:
            dims = layer.ndim
            if dims > max_dims:
                max_dims = dims

        self._max_dims = max_dims

        self._need_slider_update = True
        self._update()

    def _calc_max_shape(self):
        """Calculates the maximum shape of the contained images.
        """
        shapes = (layer.shape for layer in self.layers)
        self._max_shape = _compute_max_shape(shapes, self.max_dims)

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

    def _on_layers_change(self, event):
        """Called whenever a layer is changed.
        """
        self._child_layer_changed = True
        self._update()

    def _update_pos(self, event):
        visual = self.layers[0]._node
        tr = self._canvas.scene.node_transform(self.layers[0]._node)
        pos = tr.map(event.pos)
        self._pos = [clip(pos[1],0,self.max_shape[0]-1), clip(pos[0],0,self.max_shape[1]-1)]

    def _active_layers(self):
        from ..layers._image_layer import Image
        from ..layers._markers_layer import Markers

        top_markers = []
        for i, layer in enumerate(self.layers[::-1]):
            if layer.visible and isinstance(layer, Image):
                top_image = len(self.layers) - 1 - i
                break
            elif layer.visible and isinstance(layer, Markers):
                top_markers.append(len(self.layers) - 1 - i)
        else:
            top_image = None
        return top_image, top_markers

    def _update_statusBar(self):
        msg = '(%d, %d' % (self._pos[0], self._pos[1])
        for i in range(2,self.max_dims):
            msg = msg + ', %d' % self.indices[i]
        msg = msg + ')'

        top_image, top_markers = self._active_layers()

        index = None
        for i in top_markers:
            indices = copy(self.indices)
            indices[0] = int(self._pos[1])
            indices[1] = int(self._pos[0])
            index = self.layers[i]._selected_markers(indices)
            if index is None:
                pass
            else:
                msg = msg + ' index %d, layer %d' % (index, i)
                break

        if top_image is None:
            pass
        elif index is None:
            indices = copy(self.indices)
            indices[0] = int(self._pos[0])
            indices[1] = int(self._pos[1])
            value = self.layers[top_image]._slice_image(indices)
            if isinstance(value, ndarray):
                if isinstance(value[0], integer):
                    msg = msg + ' r %d, g %d, b %d' % (value[0], value[1], value[2])
                else:
                    msg = msg + ' r %.3f, g %.3f, b %.3f' % (value[0], value[1], value[2])
            else:
                if isinstance(value, integer):
                    msg = msg + ' value %d' % value
                else:
                    msg = msg + ' value %.3f' % value
        self._window._qt_window.statusBar().showMessage(msg)

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.
        """
        if event.pos is None:
            pass
        elif not event.is_dragging:
            self._update_pos(event)
            self._update_statusBar()

        #if event.is_dragging:
        #    print('mouse_dragging')

    def on_mouse_release(self, event):
        print('release')
        if self.interactive:
            if event.pos is None:
                pass
            else:
                if event.trail() is None:
                    accept = True
                elif len(event.trail())<2:
                    accept = True
                else:
                    accept = False
                if accept:
                    self._update_pos(event)
                    top_image, top_markers = self._active_layers()
                    print(top_markers)
                    for i in top_markers:
                        layer = self.layers[i]
                        if layer.selected:
                            indices = copy(self.indices)
                            indices[0] = int(self._pos[1])
                            indices[1] = int(self._pos[0])
                            index = layer._selected_markers(indices)
                            print(index)
                            if index is None:
                                if isinstance(layer.size, (list, ndarray)):
                                    layer._size = insert(layer.size, 0, 10)
                                layer.data = insert(layer.data, 0, [indices], axis=0)
                            else:
                                if isinstance(layer.size, (list, ndarray)):
                                    layer._size = delete(layer.size, index)
                                layer.data = delete(layer.data,index, axis=0)

        # print(event.trail())
        # print(event.is_dragging)
        # if event.is_dragging:
        #     print('drag_release')
        # else:
        #     print('normal_release')

    def on_mouse_press(self, event):
        print('mouse_click')
