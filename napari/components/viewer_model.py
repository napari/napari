import warnings

import numpy as np

from ..utils.events import EmitterGroup, Event
from ..utils.key_bindings import KeymapHandler, KeymapProvider
from ..utils.theme import palettes
from ._viewer_mouse_bindings import dims_scroll
from .add_layers_mixin import AddLayersMixin
from .axes import Axes
from .camera import Camera
from .cursor import Cursor
from .dims import Dims
from .grid import GridCanvas
from .layerlist import LayerList
from .scale_bar import ScaleBar


class ViewerModel(AddLayersMixin, KeymapHandler, KeymapProvider):
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
            reset_view=Event,
            active_layer=Event,
            palette=Event,
            layers_change=Event,
        )

        self.dims = Dims(
            ndim=None, ndisplay=ndisplay, order=order, axis_labels=axis_labels
        )

        self.layers = LayerList()
        self.camera = Camera(self.dims)
        self.cursor = Cursor()
        self.axes = Axes()
        self.scale_bar = ScaleBar()

        self._status = 'Ready'
        self._help = ''
        self._title = title

        self._interactive = True
        self._active_layer = None
        self.grid = GridCanvas()
        # 2-tuple indicating height and width
        self._canvas_size = (600, 800)
        self._palette = None
        self.theme = 'dark'

        self.grid.events.update.connect(self.reset_view)
        self.grid.events.update.connect(self._on_grid_change)
        self.dims.events.ndisplay.connect(self._update_layers)
        self.dims.events.ndisplay.connect(self.reset_view)
        self.dims.events.order.connect(self._update_layers)
        self.dims.events.order.connect(self.reset_view)
        self.dims.events.current_step.connect(self._update_layers)
        self.cursor.events.position.connect(self._on_cursor_position_change)
        self.layers.events.inserted.connect(self._on_grid_change)
        self.layers.events.removed.connect(self._on_grid_change)
        self.layers.events.reordered.connect(self._on_grid_change)
        self.layers.events.inserted.connect(self._on_layers_change)
        self.layers.events.removed.connect(self._on_layers_change)
        self.layers.events.reordered.connect(self._on_layers_change)

        self.keymap_providers = [self]

        # Hold callbacks for when mouse moves with nothing pressed
        self.mouse_move_callbacks = []
        # Hold callbacks for when mouse is pressed, dragged, and released
        self.mouse_drag_callbacks = []
        # Hold callbacks for when mouse wheel is scrolled
        self.mouse_wheel_callbacks = [dims_scroll]

        self._persisted_mouse_event = {}
        self._mouse_drag_gen = {}
        self._mouse_wheel_gen = {}

    def __str__(self):
        """Simple string representation"""
        return f'napari.Viewer: {self.title}'

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
        self.axes.background_color = self.palette['canvas']
        self.scale_bar.background_color = self.palette['canvas']
        self.events.palette()

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
        """tuple: Size of grid."""
        warnings.warn(
            (
                "The viewer.grid_size parameter is deprecated and will be removed after version 0.4.3."
                " Instead you should use viewer.grid.shape"
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.grid.shape

    @grid_size.setter
    def grid_size(self, grid_size):
        warnings.warn(
            (
                "The viewer.grid_size parameter is deprecated and will be removed after version 0.4.3."
                " Instead you should use viewer.grid.shape"
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.grid.shape = grid_size

    @property
    def grid_stride(self):
        """int: Number of layers in each grid square."""
        warnings.warn(
            (
                "The viewer.grid_stride parameter is deprecated and will be removed after version 0.4.3."
                " Instead you should use viewer.grid.stride"
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.grid.stride

    @grid_stride.setter
    def grid_stride(self, grid_stride):
        warnings.warn(
            (
                "The viewer.grid_stride parameter is deprecated and will be removed after version 0.4.3."
                " Instead you should use viewer.grid.stride"
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.grid.stride = grid_stride

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
    def active_layer(self):
        """int: index of active_layer
        """
        return self._active_layer

    @active_layer.setter
    def active_layer(self, active_layer):
        if active_layer == self.active_layer:
            return

        if self._active_layer is not None:
            self.keymap_providers.remove(self._active_layer)

        self._active_layer = active_layer

        if active_layer is not None:
            self.keymap_providers.insert(0, active_layer)

        self.events.active_layer(item=self._active_layer)

    @property
    def _sliced_extent_world(self) -> np.ndarray:
        """Extent of layers in world coordinates after slicing.

        D is either 2 or 3 depending on if the displayed data is 2D or 3D.

        Returns
        -------
        sliced_extent_world : array, shape (2, D)
        """
        if len(self.layers) == 0 and self.dims.ndim != 2:
            # If no data is present and dims model has not been reset to 0
            # than someone has passed more than two axis labels which are
            # being saved and so default values are used.
            return np.vstack(
                [np.zeros(self.dims.ndim), np.repeat(512, self.dims.ndim)]
            )
        else:
            return self.layers.extent.world[:, self.dims.displayed]

    def reset_view(self, event=None):
        """Reset the camera view."""

        extent = self._sliced_extent_world
        scene_size = extent[1] - extent[0]
        corner = extent[0]
        grid_size = list(self.grid.actual_shape(len(self.layers)))
        if len(scene_size) > len(grid_size):
            grid_size = [1] * (len(scene_size) - len(grid_size)) + grid_size
        size = np.multiply(scene_size, grid_size)
        center = np.add(corner, np.divide(size, 2))[-self.dims.ndisplay :]
        center = [0] * (self.dims.ndisplay - len(center)) + list(center)

        self.camera.center = center
        # zoom is definied as the number of canvas pixels per world pixel
        # The default value used below will zoom such that the whole field
        # of view will occupy 95% of the canvas on the most filled axis
        if np.max(size) == 0:
            self.camera.zoom = 0.95 * np.min(self._canvas_size)
        else:
            self.camera.zoom = (
                0.95 * np.min(self._canvas_size) / np.max(size[-2:])
            )
        self.camera.angles = (0, 0, 90)

        # Emit a reset view event, which is no longer used internally, but
        # which maybe useful for building on napari.
        self.events.reset_view(
            center=self.camera.center,
            zoom=self.camera.zoom,
            angles=self.camera.angles,
        )

    def _new_labels(self):
        """Create new labels layer filling full world coordinates space."""
        extent = self.layers.extent.world
        scale = self.layers.extent.step
        scene_size = extent[1] - extent[0]
        corner = extent[0]
        shape = [
            np.round(s / sc).astype('int') + 1 if s > 0 else 1
            for s, sc in zip(scene_size, scale)
        ]
        empty_labels = np.zeros(shape, dtype=int)
        self.add_labels(empty_labels, translate=np.array(corner), scale=scale)

    def _update_layers(self, event=None, layers=None):
        """Updates the contained layers.

        Parameters
        ----------
        layers : list of napari.layers.Layer, optional
            List of layers to update. If none provided updates all.
        """
        layers = layers or self.layers
        for layer in layers:
            layer._slice_dims(
                self.dims.point, self.dims.ndisplay, self.dims.order
            )

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
            self.cursor.style = 'standard'
            self.interactive = True
            self.active_layer = None
        else:
            self.status = active_layer.status
            self.help = active_layer.help
            self.cursor.style = active_layer.cursor
            self.interactive = active_layer.interactive
            self.active_layer = active_layer

    def _on_layers_change(self, event):
        if len(self.layers) == 0:
            self.dims.ndim = 2
            self.dims.reset()
        else:
            extent = self.layers.extent.world
            ss = self.layers.extent.step
            ndim = extent.shape[1]
            self.dims.ndim = ndim
            for i in range(ndim):
                self.dims.set_range(i, (extent[0, i], extent[1, i], ss[i]))
        self.events.layers_change()
        self._update_active_layer(event)

    def _update_interactive(self, event):
        """Set the viewer interactivity with the `event.interactive` bool."""
        self.interactive = event.interactive

    def _update_cursor(self, event):
        """Set the viewer cursor with the `event.cursor` string."""
        self.cursor.style = event.cursor

    def _update_cursor_size(self, event):
        """Set the viewer cursor_size with the `event.cursor_size` int."""
        self.cursor.size = event.cursor_size

    def _on_cursor_position_change(self, event):
        """Set the layer cursor position."""
        for layer in self.layers:
            layer.position = self.cursor.position

        # Update status and help bar based on active layer
        if self.active_layer is not None:
            self.status = self.active_layer.status
            self.help = self.active_layer.help

    def _on_grid_change(self, event):
        """Arrange the current layers is a 2D grid."""
        for i, layer in enumerate(self.layers[::-1]):
            i_row, i_column = self.grid.position(i, len(self.layers))
            self._subplot(layer, (i_row, i_column))

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
        warnings.warn(
            (
                "The viewer.grid_view method is deprecated and will be removed after version 0.4.3."
                " Instead you should use the viewer.grid.enabled = Turn to turn on the grid view,"
                " and viewer.grid.shape and viewer.grid.stride to set the size and stride of the"
                " grid respectively."
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.grid.stride = stride
        if n_row is None:
            n_row = -1
        if n_column is None:
            n_column = -1
        self.grid.shape = (n_row, n_column)
        self.grid.enabled = True

    def stack_view(self):
        """Arrange the current layers in a stack.
        """
        warnings.warn(
            (
                "The viewer.stack_view method is deprecated and will be removed after version 0.4.3."
                " Instead you should use the viewer.grid.enabled = False to turn off the grid view."
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.grid.enabled = False

    def _subplot(self, layer, position):
        """Shift a layer to a specified position in a 2D grid.

        Parameters
        ----------
        layer : napari.layers.Layer
            Layer that is to be moved.
        position : 2-tuple of int
            New position of layer in grid.
        size : 2-tuple of int
            Size of the grid that is being used.
        """
        extent = self._sliced_extent_world
        scene_shift = extent[1] - extent[0] + 1
        translate_2d = np.multiply(scene_shift[-2:], position)
        translate = [0] * layer.ndim
        translate[-2:] = translate_2d
        layer.translate_grid = translate

    @property
    def experimental(self):
        """Experimental commands for IPython console.

        For example run "viewer.experimental.cmds.loader.help".
        """
        from .experimental.commands import ExperimentalNamespace

        return ExperimentalNamespace(self.layers)
