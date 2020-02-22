from math import inf
import itertools
from xml.etree.ElementTree import Element, tostring

import numpy as np

from .add_layers_mixin import AddLayersMixin
from .dims import Dims
from .layerlist import LayerList
from ..utils.event import EmitterGroup, Event
from ..utils.keybindings import KeymapMixin
from ..utils.theme import palettes


class ViewerModel(AddLayersMixin, KeymapMixin):
    """Viewer containing the rendered scene, layers, and controlling elements
    including dimension sliders, and control bars for color limits.

    Parameters
    ----------
    title : string
        The title of the viewer window.
    ndisplay : {2, 3}
        Number of displayed dimensions.
    order : tuple of int
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    axis_labels = list of str
        Dimension names.

    Attributes
    ----------
    window : Window
        Parent window.
    layers : LayerList
        List of contained layers.
    dims : Dimensions
        Contains axes, indices, dimensions and sliders.
    themes : dict of str: dict of str: str
        Preset color palettes.
    """

    themes = palettes

    def __init__(
        self, title='napari', ndisplay=2, order=None, axis_labels=None
    ):
        super().__init__()

        self.events = EmitterGroup(
            source=self,
            auto_connect=True,
            status=Event,
            help=Event,
            title=Event,
            interactive=Event,
            cursor=Event,
            reset_view=Event,
            active_layer=Event,
            palette=Event,
            grid=Event,
            layers_change=Event,
        )

        self.dims = Dims(
            ndim=None, ndisplay=ndisplay, order=order, axis_labels=axis_labels
        )

        self.layers = LayerList()

        self._status = 'Ready'
        self._help = ''
        self._title = title
        self._cursor = 'standard'
        self._cursor_size = None
        self._interactive = True
        self._active_layer = None
        self._grid_size = (1, 1)
        self.grid_stride = 1

        self._palette = None
        self.theme = 'dark'

        self.dims.events.camera.connect(self.reset_view)
        self.dims.events.ndisplay.connect(self._update_layers)
        self.dims.events.order.connect(self._update_layers)
        self.dims.events.axis.connect(self._update_layers)
        self.layers.events.changed.connect(self._on_layers_change)
        self.layers.events.changed.connect(self._update_active_layer)
        self.layers.events.changed.connect(self._update_grid)

        # Hold callbacks for when mouse moves with nothing pressed
        self.mouse_move_callbacks = []
        # Hold callbacks for when mouse is pressed, dragged, and released
        self.mouse_drag_callbacks = []
        self._persisted_mouse_event = {}
        self._mouse_drag_gen = {}

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
            raise ValueError(
                f"Theme '{theme}' not found; "
                f"options are {list(self.themes)}."
            )

    @property
    def grid_size(self):
        """tuple: Size of grid
        """
        return self._grid_size

    @grid_size.setter
    def grid_size(self, grid_size):
        if np.all(self.grid_size == grid_size):
            return
        self._grid_size = grid_size
        self.reset_view()
        self.events.grid()

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

    def _scene_shape(self):
        """Get shape of currently viewed dimensions.

        Returns
        ----------
        centroid : list
            List of center coordinates of scene, length 2 or 3 if displayed
            view is 2D or 3D.
        size : list
            List of size of scene, length 2 or 3 if displayed view is 2D or 3D.
        corner : list
            List of coordinates of top left corner of scene, length 2 or 3 if
            displayed view is 2D or 3D.
        """
        # Scale the camera to the contents in the scene
        min_shape, max_shape = self._calc_bbox()
        size = np.subtract(max_shape, min_shape)
        size = [size[i] for i in self.dims.displayed]
        corner = [min_shape[i] for i in self.dims.displayed]

        return size, corner

    def reset_view(self, event=None):
        """Resets the camera's view using `event.rect` a 4-tuple of the x, y
        corner position followed by width and height of the camera
        """

        scene_size, corner = self._scene_shape()
        grid_size = list(self.grid_size)
        if len(scene_size) > len(grid_size):
            grid_size = [1] * (len(scene_size) - len(grid_size)) + grid_size
        size = np.multiply(scene_size, grid_size)
        centroid = np.add(corner, np.divide(size, 2))

        if self.dims.ndisplay == 2:
            # For a PanZoomCamera emit a 4-tuple of the rect
            corner = np.subtract(corner, np.multiply(0.05, size))[::-1]
            size = np.multiply(1.1, size)[::-1]
            rect = tuple(corner) + tuple(size)
            self.events.reset_view(rect=rect)
        else:
            # For an ArcballCamera emit the center and scale_factor
            center = centroid[::-1]
            scale_factor = 1.1 * np.max(size[-2:])
            # set initial camera angle so that it matches top layer of 2D view
            # when transitioning to 3D view
            quaternion = [np.pi / 2, 1, 0, 0]
            self.events.reset_view(
                center=center, scale_factor=scale_factor, quaternion=quaternion
            )

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

        props = {
            'xmlns': 'http://www.w3.org/2000/svg',
            'xmlns:xlink': 'http://www.w3.org/1999/xlink',
        }

        xml = Element(
            'svg',
            height=f'{shape[0]}',
            width=f'{shape[1]}',
            version='1.1',
            **props,
        )

        transform = f'translate({-min_shape[1]} {-min_shape[0]})'
        xml_transform = Element('g', transform=transform)

        for layer in self.layers:
            if layer.visible:
                xml_list = layer.to_xml_list()
                for x in xml_list:
                    xml_transform.append(x)
        xml.append(xml_transform)

        svg = (
            '<?xml version=\"1.0\" standalone=\"no\"?>\n'
            + '<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n'
            + '\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n'
            + tostring(xml, encoding='unicode', method='xml')
        )

        if file:
            # Save svg to file
            with open(file, 'w') as f:
                f.write(svg)

        return svg

    def _new_labels(self):
        if self.dims.ndim == 0:
            dims = (512, 512)
        else:
            dims = self._calc_bbox()[1]
            dims = [np.ceil(d).astype('int') if d > 0 else 1 for d in dims]
            if len(dims) < 1:
                dims = (512, 512)
        empty_labels = np.zeros(dims, dtype=int)
        self.add_labels(empty_labels)

    def _update_layers(self, event=None, layers=None):
        """Updates the contained layers.

        Parameters
        ----------
        layers : list of napari.layers.Layer, optional
            List of layers to update. If none provided updates all.
        """
        layers = layers or self.layers

        for layer in layers:
            # adjust the order of the global dims based on the number of
            # dimensions that a layer has - for example a global order of
            # [2, 1, 0, 3] -> [0, 1] for a layer that only has two dimesnions
            # or -> [1, 0, 2] for a layer with three as that corresponds to
            # the relative order of the last two and three dimensions
            # respectively
            offset = self.dims.ndim - layer.dims.ndim
            order = np.array(self.dims.order)
            if offset <= 0:
                order = list(range(-offset)) + list(order - offset)
            else:
                order = list(order[order >= offset] - offset)
            layer.dims.order = order
            layer.dims.ndisplay = self.dims.ndisplay

            # Update the point values of the layers for the dimensions that
            # the layer has
            for axis in range(layer.dims.ndim):
                point = self.dims.point[axis + offset]
                layer.dims.set_point(axis, point)

    def _toggle_theme(self):
        """Switch to next theme in list of themes
        """
        theme_names = list(self.themes.keys())
        cur_theme = theme_names.index(self.theme)
        self.theme = theme_names[(cur_theme + 1) % len(theme_names)]

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
        if len(self.layers) == 0:
            self.dims.ndim = 2
            self.dims.reset()
        else:
            layer_range = self._calc_layers_ranges()
            self.dims.ndim = len(layer_range)
            for i, r in enumerate(layer_range):
                self.dims.set_range(i, r)
        self.events.layers_change()

    def _calc_layers_ranges(self):
        """Calculates the range along each axis from all present layers.
        """

        ndims = self._calc_layers_num_dims()
        ranges = [(inf, -inf, inf)] * ndims

        for layer in self.layers:
            layer_range = layer.dims.range[::-1]
            ranges = [
                (min(a, b), max(c, d), min(e, f))
                for (a, c, e), (b, d, f) in itertools.zip_longest(
                    ranges, layer_range, fillvalue=(inf, -inf, inf)
                )
            ]

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
        if len(min_shape) == 0:
            min_shape = [0] * self.dims.ndim
            max_shape = [1] * self.dims.ndim

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

    def grid_view(self, n_row=None, n_column=None, stride=1):
        """Arrange the current layers is a 2D grid.

        Default behaviour is to make a square 2D grid.

        Parameters
        ----------
        n_row : int, optional
            Number of rows in the grid.
        n_column : int, optional
            Number of column in the grid.
        stride : int, optional
            Number of layers to place in each grid square before moving on to
            the next square. The default ordering is to place the most visible
            layer in the top left corner of the grid. A negative stride will
            cause the order in which the layers are placed in the grid to be
            reversed.
        """
        n_grid_squares = np.ceil(len(self.layers) / abs(stride)).astype(int)
        if n_row is None and n_column is None:
            n_row = np.ceil(np.sqrt(n_grid_squares)).astype(int)
            n_column = n_row
        elif n_row is None:
            n_row = np.ceil(n_grid_squares / n_column).astype(int)
        elif n_column is None:
            n_column = np.ceil(n_grid_squares / n_row).astype(int)

        n_row = max(1, n_row)
        n_column = max(1, n_column)
        self.grid_size = (n_row, n_column)
        self.grid_stride = stride
        for i, layer in enumerate(self.layers):
            if stride > 0:
                adj_i = len(self.layers) - i - 1
            else:
                adj_i = i
            adj_i = adj_i // abs(stride)
            adj_i = adj_i % (n_row * n_column)
            i_row = adj_i // n_column
            i_column = adj_i % n_column
            self._subplot(layer, (i_row, i_column))

    def stack_view(self):
        """Arrange the current layers is a stack.
        """
        self.grid_view(n_row=1, n_column=1, stride=1)

    def _update_grid(self, event=None):
        """Update grid with current grid values.
        """
        self.grid_view(
            n_row=self.grid_size[0],
            n_column=self.grid_size[1],
            stride=self.grid_stride,
        )

    def _subplot(self, layer, position):
        """Shift a layer to a specified position in a 2D grid.

        Parameters
        ----------
        layer : napar.layers.Layer
            Layer that is to be moved.
        position : 2-tuple of int
            New position of layer in grid.
        size : 2-tuple of int
            Size of the grid that is being used.
        """
        scene_size, corner = self._scene_shape()
        translate_2d = np.multiply(scene_size[-2:], position)
        translate = [0] * layer.ndim
        translate[-2:] = translate_2d
        layer.translate_grid = translate
