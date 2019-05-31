import numpy as np
from math import inf
from copy import copy
from itertools import zip_longest
from xml.etree.ElementTree import Element, tostring

from ...util.event import EmitterGroup, Event
from ...util.theme import palettes
from .._dims import Dims


class ViewerModel:
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
    key_bindings : dict of string: callable
        Custom key bindings. The dictionary key is a string containing the key
        pressed and the value is the function to be bound to the key event.
        The function should accept the viewer object as an input argument.
        These key bindings are executed instead of any layer specific key
        bindings.
    themes : dict of str: dict of str: str
        Preset color palettes.
    """
    themes = palettes

    def __init__(self, title='napari'):
        super().__init__()
        from .._layers_list import LayersList

        self.events = EmitterGroup(source=self,
                                   auto_connect=True,
                                   status=Event,
                                   help=Event,
                                   title=Event,
                                   interactive=Event,
                                   cursor=Event,
                                   reset_view=Event,
                                   active_layer=Event,
                                   palette=Event)

        # Initial dimension must be set to at least the number of visible
        # dimensions of the viewer
        self.dims = Dims(2)
        self.dims._set_2d_viewing()

        self.layers = LayersList()

        self._status = 'Ready'
        self._help = ''
        self._title = title
        self._cursor = 'standard'
        self._cursor_size = None
        self._interactive = True
        self._active_layer = None
        self.key_bindings = {}

        self._palette = None
        self.theme = 'dark'

        # TODO: this should be eventually removed!
        # attached by QtViewer when it is constructed by the model
        self._view = None

        self.dims.events.axis.connect(lambda e: self._update_layers())
        self.layers.events.added.connect(self._on_layers_change)
        self.layers.events.removed.connect(self._on_layers_change)
        self.layers.events.added.connect(self._update_active_layer)
        self.layers.events.removed.connect(self._update_active_layer)
        self.layers.events.reordered.connect(self._update_active_layer)

    @property
    def palette(self):
        """dict of str: str : Color palette with which to style the viewer.
        """
        return self._palette

    @palette.setter
    def palette(self, palette):
        if palette == self.palette:
            return

        self._palette = palette
        self.events.palette(palette=palette)

    @property
    def theme(self):
        """string or None : Preset color palette.
        """
        for theme, palette in self.themes.items():
            if palette == self.palette:
                return theme

    @theme.setter
    def theme(self, theme):
        if theme == self.theme:
            return

        try:
            self.palette = self.themes[theme]
        except KeyError:
            raise ValueError(f"Theme '{theme}' not found; "
                             f"options are {list(self.themes)}.")

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
        self._interactive = interactive
        self.events.interactive()

    @property
    def cursor(self):
        """string: String identifying cursor displayed over canvas.
        """
        return self._cursor

    @cursor.setter
    def cursor(self, cursor):
        if cursor == self.cursor:
            return
        self._cursor = cursor
        self.events.cursor()

    @property
    def cursor_size(self):
        """int | None: Size of cursor if custom. None is yields default size
        """
        return self._cursor_size

    @cursor_size.setter
    def cursor_size(self, cursor_size):
        if cursor_size == self.cursor_size:
            return
        self._cursor_size = cursor_size
        self.events.cursor()

    @property
    def active_layer(self):
        """int: index of active_layer
        """
        return self._active_layer

    @active_layer.setter
    def active_layer(self, active_layer):
        if active_layer == self.active_layer:
            return
        self._active_layer = active_layer
        self.events.active_layer(item=self._active_layer)

    def reset_view(self):
        """Resets the camera's view.
        """
        self.events.reset_view()

    def to_svg(self, file=None, view_box=None):
        """Convert the viewer state to an SVG. Non visible layers will be
        ignored.

        Parameters
        ----------
        file : path-like object, optional
            An object representing a file system path. A path-like object is
            either a str or bytes object representing a path, or an object
            implementing the `os.PathLike` protocol. If passed the svg will be
            written to this file
        view_box : 4-tuple, optional
            View box of SVG canvas to be generated specified as `min-x`,
            `min-y`, `width` and `height`. If not specified, calculated
            from the last two dimensions of the view.

        Returns
        ----------
        svg : string
            SVG representation of the currently viewed layers.
        """

        if view_box is None:
            min_shape, max_shape = self._calc_bbox()
            min_shape = min_shape[-2:]
            max_shape = max_shape[-2:]
            shape = np.subtract(max_shape, min_shape)
        else:
            shape = view_box[2:]
            min_shape = view_box[:2]

        props = {'xmlns': 'http://www.w3.org/2000/svg',
                 'xmlns:xlink': 'http://www.w3.org/1999/xlink'}

        xml = Element('svg', height=f'{shape[0]}', width=f'{shape[1]}',
                      version='1.1', **props)

        transform = f'translate({-min_shape[1]} {-min_shape[0]})'
        xml_transform = Element('g', transform=transform)

        for layer in self.layers:
            if layer.visible:
                xml_list = layer.to_xml_list()
                for x in xml_list:
                    xml_transform.append(x)
        xml.append(xml_transform)

        svg = ('<?xml version=\"1.0\" standalone=\"no\"?>\n' +
               '<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n' +
               '\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n' +
               tostring(xml, encoding='unicode', method='xml'))

        if file:
            # Save svg to file
            with open(file, 'w') as f:
                f.write(svg)

        return svg

    def add_layer(self, layer):
        """Adds a layer to the viewer.

        Parameters
        ----------
        layer : Layer
            Layer to add.
        """
        layer.events.select.connect(self._update_active_layer)
        layer.events.deselect.connect(self._update_active_layer)
        layer.events.status.connect(self._update_status)
        layer.events.help.connect(self._update_help)
        layer.events.interactive.connect(self._update_interactive)
        layer.events.cursor.connect(self._update_cursor)
        layer.events.cursor_size.connect(self._update_cursor_size)
        layer.events.name.connect(self._update_name)
        layer.events.data.connect(self._on_layers_change)

        layer.parent = self._view
        self.layers.append(layer)
        layer.indices = self.dims.indices

        if len(self.layers) == 1:
            self.reset_view()

        self.layers.unselect_all(ignore=layer)

    def _new_markers(self):
        if self.dims.ndim == 0:
            empty_markers = np.empty((0, 2))
        else:
            empty_markers = np.empty((0, self.dims.ndim))
        self.add_markers(empty_markers)

    def _new_shapes(self):
        layer = self.add_shapes([], ndim=self.dims.ndim)

    def _new_labels(self):
        if self.dims.ndim == 0:
            empty_labels = np.zeros((512, 512), dtype=int)
        else:
            empty_labels = np.zeros(self._calc_bbox()[1], dtype=int)
        self.add_labels(empty_labels)

    def _update_layers(self):
        """Updates the contained layers.
        """
        for layer in self.layers:
            layer.indices = self.dims.indices

    def _update_active_layer(self, event):
        """Set the active layer by iterating over the layers list and
        finding the first selected layer. If multiple layers are selected the
        iteration stops and the active layer is set to be None

        Parameters
        ----------
        event : Event
            No Event parameters are used
        """
        # iteration goes backwards to find top most selected layer if any
        # if multiple layers are selected sets the active layer to None
        active_layer = None
        for layer in self.layers:
            if active_layer is None and layer.selected:
                active_layer = layer
            elif active_layer is not None and layer.selected:
                active_layer = None
                break

        if active_layer is None:
            self.status = 'Ready'
            self.help = ''
            self.cursor = 'standard'
            self.interactive = True
            self.active_layer = None
        else:
            self.status = active_layer.status
            self.help = active_layer.help
            self.cursor = active_layer.cursor
            self.interactive = active_layer.interactive
            self.active_layer = active_layer

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

    def _calc_bbox(self):
        """Calculates the bounding box of all displayed layers.
        This assumes that all layers are stacked.
        """

        min_shape = []
        max_shape = []
        for min, max, step in self._calc_layers_ranges():
            min_shape.append(min)
            max_shape.append(max)

        return min_shape, max_shape

    def _calc_layers_num_dims(self):
        """Calculates the number of maximum dimensions in the contained images.
        """
        max_dims = 0
        for layer in self.layers:
            dims = layer.ndim
            if dims > max_dims:
                max_dims = dims

        return max_dims

    def _update_status(self, event):
        """Set the viewer status with the `event.status` string."""
        self.status = event.status

    def _update_help(self, event):
        """Set the viewer help with the `event.help` string."""
        self.help = event.help

    def _update_interactive(self, event):
        """Set the viewer interactivity with the `event.interactive` bool."""
        self.interactive = event.interactive

    def _update_cursor(self, event):
        """Set the viewer cursor with the `event.cursor` string."""
        self.cursor = event.cursor

    def _update_cursor_size(self, event):
        """Set the viewer cursor_size with the `event.cursor_size` int."""
        self.cursor_size = event.cursor_size

    def _update_name(self, event):
        """Coerce name of the layer in `event.layer`."""
        layer = event.source
        layer.name = self.layers._coerce_name(layer.name, layer)
