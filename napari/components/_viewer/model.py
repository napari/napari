import numpy as np
from math import inf
from copy import copy
from itertools import zip_longest

from ...util.event import EmitterGroup, Event


class Viewer:
    """Viewer containing the rendered scene, layers, and controlling elements
    including dimension sliders, and control bars for color limits.

    Attributes
    ----------
    window : Window
        Parent window.
    layers : LayersList
        List of contained layers.
    dims : Dimensions
        Contains axes, indices, dimensions and sliders.
    camera : vispy.scene.Camera
        Viewer camera.
    key_bindings : dict of string: callable
        Custom key bindings. The dictionary key is a string containing the key
        pressed and the value is the function to be bound to the key event.
        The function should accept the viewer object as an input argument.
        These key bindings are executed instead of any layer specific key
        bindings.
    """
    def __init__(self, title='napari'):
        super().__init__()
        from .._layers_list import LayersList
        from .._dims import Dims

        self.events = EmitterGroup(source=self,
                                   auto_connect=True,
                                   status=Event,
                                   help=Event,
                                   title=Event,
                                   active_markers=Event)

        # Initial dimension must be set to at least the number of visible
        # dimensions of the viewer
        self.dims = Dims(2)
        self.dims._set_2d_viewing()

        self.layers = LayersList(self)

        self._status = 'Ready'
        self._help = ''
        self._title = title
        self._cursor = 'standard'
        self._cursor_size = None
        self._interactive = True
        self._top = None
        self.key_bindings = {}

        # TODO: this should be eventually removed!
        # initialised by QtViewer when it is constructed by the model
        self._qtviewer = None

        self.dims.events.axis.connect(lambda e: self._update_layers())

    @property
    def _canvas(self):
        return self._qtviewer.canvas

    @property
    def _view(self):
        return self._qtviewer.view

    @property
    def camera(self):
        """vispy.scene.Camera: Viewer camera.
        """
        return self._view.camera

    @property
    def status(self):
        """string: Status string
        """
        return self._status

    @status.setter
    def status(self, status):
        if status == self.status:
            return
        self._status = status
        self.events.status(text=self._status)

    @property
    def help(self):
        """string: String that can be displayed to the
        user in the status bar with helpful usage tips.
        """
        return self._help

    @help.setter
    def help(self, help):
        if help == self.help:
            return
        self._help = help
        self.events.help(text=self._help)

    @property
    def title(self):
        """string: String that is displayed in window title.
        """
        return self._title

    @title.setter
    def title(self, title):
        if title == self.title:
            return
        self._title = title
        self.events.title(text=self._title)

    @property
    def interactive(self):
        """bool: Determines if canvas pan/zoom interactivity is enabled or not.
        """
        return self._interactive

    @interactive.setter
    def interactive(self, interactive):
        if interactive == self.interactive:
            return
        self._qtviewer.view.interactive = interactive
        self._interactive = interactive

    @property
    def cursor(self):
        """string: String identifying cursor displayed over canvas.
        """
        return self._cursor

    @cursor.setter
    def cursor(self, cursor):
        if cursor == self.cursor:
            return
        self._qtviewer.set_cursor(cursor, self.cursor_size)
        self._cursor = cursor

    @property
    def cursor_size(self):
        """int | None: Size of cursor if custom. None is yields default size
        """
        return self._cursor_size

    @cursor_size.setter
    def cursor_size(self, cursor_size):
        if cursor_size == self.cursor_size:
            return
        self._qtviewer.set_cursor(self.cursor, cursor_size)
        self._cursor_size = cursor_size

    @property
    def active_markers(self):
        """int: index of active_markers
        """
        return self._active_markers

    @active_markers.setter
    def active_markers(self, active_markers):
        if active_markers == self.active_markers:
            return
        self._active_markers = active_markers
        self.events.active_markers(index=self._active_markers)

    def reset_view(self):
        """Resets the camera's view.
        """
        self._qtviewer.view.camera.set_range()

    def screenshot(self, region=None, size=None, bgcolor=None):
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

        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.
        """
        return self.canvas.render(region, size, bgcolor)

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
        if self.dims.ndim == 0:
            empty_markers = np.empty((0, 2))
        else:
            empty_markers = np.empty((0, self.dims.ndim))
        self.add_markers(empty_markers)

    def _new_shapes(self):
        self.add_shapes([])

    def _new_labels(self):
        if self.dims.ndim == 0:
            empty_labels = np.zeros((512, 512), dtype=int)
        else:
            empty_labels = np.zeros(self._calc_max_shape(), dtype=int)
        self.add_labels(empty_labels)

    def _update_layers(self):
        """Updates the contained layers.
        """
        for layer in self.layers:
            layer._set_view_slice(self.dims.indices)

    def _update_layer_selection(self, event):
        # iteration goes backwards to find top most selected layer if any
        for layer in self.layers[::-1]:
            if layer.selected:
                self._qtviewer.control_panel.display(layer)
                self.status = layer.status
                self.help = layer.help
                self.cursor = layer.cursor
                self.interactive = layer.interactive
                self._top = layer
                break
        else:
            self._qtviewer.control_panel.display(None)
            self.status = 'Ready'
            self.help = ''
            self.cursor = 'standard'
            self.interactive = True
            self._top = None
        self._qtviewer.canvas.native.setFocus()

    def _on_layers_change(self, event):
        self.dims.range = self._calc_layers_ranges()

    def _calc_layers_ranges(self):
        """Calculates the range along each axis from all present layers.
        """

        ndims = self._calc_layers_num_dims()
        ranges = [(inf, -inf, inf)]*ndims

        for layer in self.layers:
            layer_range = layer.range[::-1]
            ranges = [(min(a, b), max(c, d), min(e, f)) for
                      (a, c, e), (b, d, f) in zip_longest(ranges, layer_range,
                      fillvalue=(inf, -inf, inf))]

        return ranges[::-1]

    def _calc_max_shape(self):
        """Calculates the max shape of all displayed layers.
        This assumes that all layers are stacked.
        TODO: This is a temporary workaround until refactor is done
        this method should not be used but instead '_calc_layers_ranges' should
        be called.
        """

        max_shape = [max-min for min, max, step in self._calc_layers_ranges()]

        return max_shape

    def _calc_layers_num_dims(self):
        """Calculates the number of maximum dimensions in the contained images.
        """
        max_dims = 0
        for layer in self.layers:
            dims = layer.ndim
            if dims > max_dims:
                max_dims = dims

        return max_dims
