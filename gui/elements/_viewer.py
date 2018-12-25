from numpy import clip, integer, ndarray, append, insert, delete, empty
from copy import copy

from .qt import QtViewer

class Viewer:
    """Viewer containing the rendered scene, layers, and controlling elements
    including dimension sliders, and control bars for color limits.

    Attributes
    ----------
    window : Window
        Parent window.
    layers : LayerList
        List of contained layers.
    dimensions : Dimensions
        Contains axes, indices, dimensions and sliders.
    control_bars : ControlBars
        Contains control bar sliders.
    camera : vispy.scene.Camera
        Viewer camera.
    """

    def __init__(self):
        super().__init__()
        from ._layer_list import LayerList
        from ._control_bars import ControlBars
        from ._dimensions import Dimensions

        self.dimensions = Dimensions(self)
        self.layers = LayerList(self)
        self.control_bars = ControlBars(self)

        self._update = self.dimensions._update

        self.annotation = False
        self._annotation_history = False
        self._active_image = None
        self._active_markers = None
        self._visible_markers = []

        self._status = 'Ready'
        self._help = ''

        self._qt = QtViewer(self)

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

    def reset_view(self):
        """Resets the camera's view.
        """
        self.camera.set_range()

    def screenshot(self, region=None, size=None, bgcolor=None, crop=None):
        """Render the scene to an offscreen buffer and return the image array.

        Parameters
        ----------
        region : tuple | None
            Specifies the region of the canvas to render. Format is
            (x, y, w, h). By default, the entire canvas is rendered.
        size : tuple | None
            Specifies the size of the image array to return. If no size is
            given, then the size of the *region* is used, multiplied by the
            pixel scaling factor of the canvas (see `pixel_scale`). This
            argument allows the scene to be rendered at resolutions different
            from the native canvas resolution.
        bgcolor : instance of Color | None
            The background color to use.
        crop : array-like | None
            If specified it determines the pixels read from the framebuffer.
            In the format (x, y, w, h), relative to the region being rendered.
        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.

        """
        return self._canvas.render(region=None, size=None, bgcolor=None, crop=None)

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

    def _new_markers(self):
        if self.dimensions.max_dims == 0:
            empty_markers = empty((0, 2))
        else:
            empty_markers = empty((0, self.dimensions.max_dims))
        self.add_markers(empty_markers)

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

    def _update_layers(self):
        """Updates the contained layers.
        """
        for layer in self.layers:
            layer._set_view_slice(self.dimensions.indices)
        self.dimensions._update_index(None)
        self._update_status_bar()

    def _on_layers_change(self, event):
        """Called whenever a layer is changed.
        """
        self.dimensions._child_layer_changed = True
        self.dimensions._update()

    def _set_annotation(self, bool):
        if bool:
            self.annotation = True
            self._qt.view.interactive = False
            if self._active_markers:
                self._qt.canvas.native.setCursor(self._qt._cursors['cross'])
            else:
                self._qt.canvas.native.setCursor(self._qt._cursors['disabled'])
            self._help = 'hold <space> to pan/zoom'
        else:
            self.annotation = False
            self._qt.view.interactive = True
            self._qt.canvas.native.setCursor(self._qt._cursors['standard'])
            self._help = ''
        self._qt.helpChanged.emit(self._help)
        self._update_status_bar()

    def _update_active_layers(self, event):
        from ..layers._image_layer import Image
        from ..layers._markers_layer import Markers
        top_markers = []
        for i, layer in enumerate(self.layers[::-1]):
            if layer.visible and isinstance(layer, Image):
                top_image = len(self.layers) - 1 - i
                break
            elif layer.visible and isinstance(layer, Markers):
                if self.dimensions._index is None:
                    pass
                else:
                    top_markers.append(len(self.layers) - 1 - i)
                    coord = [self.dimensions._index[1],self.dimensions._index[0],*self.dimensions._index[2:]]
                    layer._set_selected_markers(coord)
        else:
            top_image = None

        active_markers = None
        for i in top_markers:
            if self.layers[i].selected:
                active_markers = i
                break

        self._active_image = top_image
        self._visible_markers = top_markers
        self._active_markers = active_markers
        self.control_bars.clim_slider_update()

    def _reset(self):
        self._set_annotation(self.annotation)
        self._status = 'Ready'
        self.emit_status()
        self.control_bars.clim_slider_update()

    def _update_status_bar(self):
        msg = f'{self.dimensions._index}'

        index = None
        for i in self._visible_markers:
            index = self.layers[i]._selected_markers
            if index is None:
                pass
            else:
                msg = msg + ', %s, index %d' % (self.layers[i].name, index)
                break

        if self._active_image is None:
            pass
        elif index is None:
            msg = msg + ', %s' % self.layers[self._active_image].name
            value = self.layers[self._active_image]._slice_image(self.dimensions._index)
            msg = msg + ', value '
            if isinstance(value, ndarray):
                if isinstance(value[0], integer):
                    msg = msg + '(%d, %d, %d)' % (value[0], value[1], value[2])
                else:
                    msg = msg + '(%.3f, %.3f, %.3f)' % (value[0], value[1], value[2])
            else:
                if isinstance(value, integer):
                    msg = msg + '%d' % value
                else:
                    msg = msg + '%.3f' % value
        self._status = msg
        self.emit_status()

    def emit_status(self):
        self._qt.statusChanged.emit(self._status)
