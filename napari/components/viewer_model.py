import inspect
import itertools
import os
import warnings
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import numpy as np

from .. import layers
from ..layers.image._image_utils import guess_labels
from ..layers.utils.stack_utils import split_channels
from ..plugins.io import read_data_with_plugins
from ..types import FullLayerData, LayerData
from ..utils import config
from ..utils._register import create_func as create_add_method
from ..utils.colormaps import ensure_colormap
from ..utils.events import EmitterGroup, Event, disconnect_events
from ..utils.key_bindings import KeymapHandler, KeymapProvider
from ..utils.misc import is_sequence
from ..utils.theme import palettes
from ._viewer_mouse_bindings import dims_scroll
from .axes import Axes
from .camera import Camera
from .cursor import Cursor
from .dims import Dims
from .grid import GridCanvas
from .layerlist import LayerList
from .scale_bar import ScaleBar


class ViewerModel(KeymapHandler, KeymapProvider):
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

    def __init__(self, title='napari', ndisplay=2, order=(), axis_labels=()):
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
            ndisplay=ndisplay, order=order, axis_labels=axis_labels
        )

        self.layers = LayerList()
        self.camera = Camera()
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

        self.grid.events.connect(self.reset_view)
        self.grid.events.connect(self._on_grid_change)
        self.dims.events.ndisplay.connect(self._update_layers)
        self.dims.events.ndisplay.connect(self.reset_view)
        self.dims.events.order.connect(self._update_layers)
        self.dims.events.order.connect(self.reset_view)
        self.dims.events.current_step.connect(self._update_layers)
        self.cursor.events.position.connect(self._on_cursor_position_change)
        self.layers.events.inserted.connect(self._on_add_layer)
        self.layers.events.removed.connect(self._on_remove_layer)
        self.layers.events.reordered.connect(self._on_grid_change)
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

        # Only created if NAPARI_MON is enabled.
        self._remote_commands = _create_remote_commands(self.layers)

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
                "The viewer.grid_size parameter is deprecated and will be removed after version 0.4.4."
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
                "The viewer.grid_size parameter is deprecated and will be removed after version 0.4.4."
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
                "The viewer.grid_stride parameter is deprecated and will be removed after version 0.4.4."
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
                "The viewer.grid_stride parameter is deprecated and will be removed after version 0.4.4."
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
            self.cursor.size = active_layer.cursor_size
            self.interactive = active_layer.interactive
            self.active_layer = active_layer

    def _on_layers_change(self, event):
        if len(self.layers) == 0:
            self.dims.ndim = 2
            self.dims.reset()
        else:
            extent = self.layers.extent
            world = extent.world
            ss = extent.step
            ndim = world.shape[1]
            self.dims.ndim = ndim
            for i in range(ndim):
                self.dims.set_range(i, (world[0, i], world[1, i], ss[i]))
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
        extent = self._sliced_extent_world
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            i_row, i_column = self.grid.position(n_layers - 1 - i, n_layers)
            self._subplot(layer, (i_row, i_column), extent)

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
                "The viewer.grid_view method is deprecated and will be removed after version 0.4.4."
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
                "The viewer.stack_view method is deprecated and will be removed after version 0.4.4."
                " Instead you should use the viewer.grid.enabled = False to turn off the grid view."
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.grid.enabled = False

    def _subplot(self, layer, position, extent):
        """Shift a layer to a specified position in a 2D grid.

        Parameters
        ----------
        layer : napari.layers.Layer
            Layer that is to be moved.
        position : 2-tuple of int
            New position of layer in grid.
        extent : array, shape (2, D)
            Extent of the world.
        """
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

    def _on_add_layer(self, event):
        """Connect new layer events.

        Parameters
        ----------
        event : :class:`napari.layers.Layer`
            Layer to add.
        """
        layer = event.value

        # Coerce name into being unique and connect event to ensure uniqueness
        layer.name = self.layers._coerce_name(layer.name, layer)

        # Connect individual layer events to viewer events
        layer.events.select.connect(self._update_active_layer)
        layer.events.deselect.connect(self._update_active_layer)
        layer.events.interactive.connect(self._update_interactive)
        layer.events.cursor.connect(self._update_cursor)
        layer.events.cursor_size.connect(self._update_cursor_size)
        layer.events.data.connect(self._on_layers_change)
        layer.events.scale.connect(self._on_layers_change)
        layer.events.translate.connect(self._on_layers_change)
        layer.events.rotate.connect(self._on_layers_change)
        layer.events.shear.connect(self._on_layers_change)
        layer.events.affine.connect(self._on_layers_change)
        layer.events.name.connect(self.layers._update_name)

        # For the labels layer we need to reset the undo/ redo
        # history whenever the displayed slice changes. Once
        # we have full undo/ redo functionality, this can be
        # dropped.
        if hasattr(layer, '_reset_history'):
            self.dims.events.ndisplay.connect(layer._reset_history)
            self.dims.events.order.connect(layer._reset_history)
            self.dims.events.current_step.connect(layer._reset_history)

        # Make layer selected and unselect all others
        layer.selected = True
        self.layers.unselect_all(ignore=layer)

        # Update dims and grid model
        self._on_layers_change(None)
        self._on_grid_change(None)
        # Slice current layer based on dims
        self._update_layers(layers=[layer])

        if len(self.layers) == 1:
            self.reset_view()

    def _on_remove_layer(self, event):
        """Disconnect old layer events.

        Parameters
        ----------
        layer : :class:`napari.layers.Layer`
            Layer to add.

        Returns
        -------
        layer : :class:`napari.layers.Layer` or list
            The layer that was added (same as input).
        """
        layer = event.value

        # Disconnect all connections from layer
        disconnect_events(layer.events, self)
        disconnect_events(layer.events, self.layers)

        # For the labels layer disconnect history resets
        if hasattr(layer, '_reset_history'):
            self.dims.events.ndisplay.disconnect(layer._reset_history)
            self.dims.events.order.disconnect(layer._reset_history)
            self.dims.events.current_step.disconnect(layer._reset_history)
        self._on_layers_change(None)
        self._on_grid_change(None)

    def add_layer(self, layer: layers.Layer) -> layers.Layer:
        """Add a layer to the viewer.

        Parameters
        ----------
        layer : :class:`napari.layers.Layer`
            Layer to add.

        Returns
        -------
        layer : :class:`napari.layers.Layer` or list
            The layer that was added (same as input).
        """
        # Adding additional functionality inside `add_layer`
        # should be avoided to keep full functionality
        # from adding a layer through the `layers.append`
        # method
        self.layers.append(layer)
        return layer

    def add_image(
        self,
        data=None,
        *,
        channel_axis=None,
        rgb=None,
        colormap=None,
        contrast_limits=None,
        gamma=1,
        interpolation='nearest',
        rendering='mip',
        iso_threshold=0.5,
        attenuation=0.05,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=1,
        blending=None,
        visible=True,
        multiscale=None,
    ) -> Union[layers.Image, List[layers.Image]]:
        """Add an image layer to the layer list.

        Parameters
        ----------
        data : array or list of array
            Image data. Can be N dimensional. If the last dimension has length
            3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a
            list and arrays are decreasing in shape then the data is treated as
            a multiscale image.
        channel_axis : int, optional
            Axis to expand image along.  If provided, each channel in the data
            will be added as an individual image layer.  In channel_axis mode,
            all other parameters MAY be provided as lists, and the Nth value
            will be applied to the Nth channel in the data.  If a single value
            is provided, it will be broadcast to all Layers.
        rgb : bool or list
            Whether the image is rgb RGB or RGBA. If not specified by user and
            the last dimension of the data has length 3 or 4 it will be set as
            `True`. If `False` the image is interpreted as a luminance image.
            If a list then must be same length as the axis that is being
            expanded as channels.
        colormap : str, napari.utils.Colormap, tuple, dict, list
            Colormaps to use for luminance images. If a string must be the name
            of a supported colormap from vispy or matplotlib. If a tuple the
            first value must be a string to assign as a name to a colormap and
            the second item must be a Colormap. If a dict the key must be a
            string to assign as a name to a colormap and the value must be a
            Colormap. If a list then must be same length as the axis that is
            being expanded as channels, and each colormap is applied to each
            new image layer.
        contrast_limits : list (2,)
            Color limits to be used for determining the colormap bounds for
            luminance images. If not passed is calculated as the min and max of
            the image. If list of lists then must be same length as the axis
            that is being expanded and then each colormap is applied to each
            image.
        gamma : list, float
            Gamma correction for determining colormap linearity. Defaults to 1.
            If a list then must be same length as the axis that is being
            expanded as channels.
        interpolation : str or list
            Interpolation mode used by vispy. Must be one of our supported
            modes. If a list then must be same length as the axis that is being
            expanded as channels.
        rendering : str or list
            Rendering mode used by vispy. Must be one of our supported
            modes. If a list then must be same length as the axis that is being
            expanded as channels.
        iso_threshold : float or list
            Threshold for isosurface. If a list then must be same length as the
            axis that is being expanded as channels.
        attenuation : float or list
            Attenuation rate for attenuated maximum intensity projection. If a
            list then must be same length as the axis that is being expanded as
            channels.
        name : str or list of str
            Name of the layer.  If a list then must be same length as the axis
            that is being expanded as channels.
        metadata : dict or list of dict
            Layer metadata. If a list then must be a list of dicts with the
            same length as the axis that is being expanded as channels.
        scale : tuple of float or list
            Scale factors for the layer. If a list then must be a list of
            tuples of float with the same length as the axis that is being
            expanded as channels.
        translate : tuple of float or list
            Translation values for the layer. If a list then must be a list of
            tuples of float with the same length as the axis that is being
            expanded as channels.
        rotate : float, 3-tuple of float, n-D array or list.
            If a float convert into a 2D rotation matrix using that value as an
            angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
            pitch, roll convention. Otherwise assume an nD rotation. Angles are
            assumed to be in degrees. They can be converted from radians with
            np.degrees if needed. If a list then must have same length as
            the axis that is being expanded as channels.
        shear : 1-D array or list.
            A vector of shear values for an upper triangular n-D shear matrix.
            If a list then must have same length as the axis that is being
            expanded as channels.
        affine: n-D array or napari.utils.transforms.Affine
            (N+1, N+1) affine transformation matrix in homogeneous coordinates.
            The first (N, N) entries correspond to a linear transform and
            the final column is a lenght N translation vector and a 1 or a napari
            AffineTransform object. If provided then translate, scale, rotate, and
            shear values are ignored.
        opacity : float or list
            Opacity of the layer visual, between 0.0 and 1.0.  If a list then
            must be same length as the axis that is being expanded as channels.
        blending : str or list
            One of a list of preset blending modes that determines how RGB and
            alpha values of the layer visual get mixed. Allowed values are
            {'opaque', 'translucent', and 'additive'}. If a list then
            must be same length as the axis that is being expanded as channels.
        visible : bool or list of bool
            Whether the layer visual is currently being displayed.
            If a list then must be same length as the axis that is
            being expanded as channels.
        multiscale : bool
            Whether the data is a multiscale image or not. Multiscale data is
            represented by a list of array like image data. If not specified by
            the user and if the data is a list of arrays that decrease in shape
            then it will be taken to be multiscale. The first image in the list
            should be the largest.

        Returns
        -------
        layer : :class:`napari.layers.Image` or list
            The newly-created image layer or list of image layers.
        """

        if colormap is not None:
            # standardize colormap argument(s) to Colormaps, and make sure they
            # are in AVAILABLE_COLORMAPS.  This will raise one of many various
            # errors if the colormap argument is invalid.  See
            # ensure_colormap for details
            if isinstance(colormap, list):
                colormap = [ensure_colormap(c) for c in colormap]
            else:
                colormap = ensure_colormap(colormap)

        # doing this here for IDE/console autocompletion in add_image function.
        kwargs = {
            'rgb': rgb,
            'colormap': colormap,
            'contrast_limits': contrast_limits,
            'gamma': gamma,
            'interpolation': interpolation,
            'rendering': rendering,
            'iso_threshold': iso_threshold,
            'attenuation': attenuation,
            'name': name,
            'metadata': metadata,
            'scale': scale,
            'translate': translate,
            'rotate': rotate,
            'shear': shear,
            'affine': affine,
            'opacity': opacity,
            'blending': blending,
            'visible': visible,
            'multiscale': multiscale,
        }

        # these arguments are *already* iterables in the single-channel case.
        iterable_kwargs = {
            'scale',
            'translate',
            'rotate',
            'shear',
            'affine',
            'contrast_limits',
            'metadata',
        }

        # Image or OctreeImage.
        image_class = _get_image_class()

        if channel_axis is None:
            kwargs['colormap'] = kwargs['colormap'] or 'gray'
            kwargs['blending'] = kwargs['blending'] or 'translucent'
            # Helpful message if someone tries to add mulit-channel kwargs,
            # but forget the channel_axis arg
            for k, v in kwargs.items():
                if k not in iterable_kwargs and is_sequence(v):
                    raise TypeError(
                        f"Received sequence for argument '{k}', "
                        "did you mean to specify a 'channel_axis'? "
                    )
            layer = image_class(data, **kwargs)
            self.layers.append(layer)

            return layer
        else:
            layerdata_list = split_channels(data, channel_axis, **kwargs)

            layer_list = list()
            for image, i_kwargs, _ in layerdata_list:
                layer = image_class(image, **i_kwargs)
                self.layers.append(layer)
                layer_list.append(layer)

            return layer_list

    def open(
        self,
        path: Union[str, Sequence[str]],
        *,
        stack: bool = False,
        plugin: Optional[str] = None,
        layer_type: Optional[str] = None,
        **kwargs,
    ) -> List[layers.Layer]:
        """Open a path or list of paths with plugins, and add layers to viewer.

        A list of paths will be handed one-by-one to the napari_get_reader hook
        if stack is False, otherwise the full list is passed to each plugin
        hook.

        Parameters
        ----------
        path : str or list of str
            A filepath, directory, or URL (or a list of any) to open.
        stack : bool, optional
            If a list of strings is passed and ``stack`` is ``True``, then the
            entire list will be passed to plugins.  It is then up to individual
            plugins to know how to handle a list of paths.  If ``stack`` is
            ``False``, then the ``path`` list is broken up and passed to plugin
            readers one by one.  by default False.
        plugin : str, optional
            Name of a plugin to use.  If provided, will force ``path`` to be
            read with the specified ``plugin``.  If the requested plugin cannot
            read ``path``, an exception will be raised.
        layer_type : str, optional
            If provided, will force data read from ``path`` to be passed to the
            corresponding ``add_<layer_type>`` method (along with any
            additional) ``kwargs`` provided to this function.  This *may*
            result in exceptions if the data returned from the path is not
            compatible with the layer_type.
        **kwargs
            All other keyword arguments will be passed on to the respective
            ``add_layer`` method.

        Returns
        -------
        layers : list
            A list of any layers that were added to the viewer.
        """
        paths = [path] if isinstance(path, str) else path
        paths = [os.fspath(path) for path in paths]  # PathObjects -> str
        if not isinstance(paths, (tuple, list)):
            raise ValueError(
                "'path' argument must be a string, list, or tuple"
            )

        if stack:
            return self._add_layers_with_plugins(
                paths, kwargs, plugin=plugin, layer_type=layer_type
            )

        added: List[layers.Layer] = []  # for layers that get added
        for _path in paths:
            added.extend(
                self._add_layers_with_plugins(
                    _path, kwargs, plugin=plugin, layer_type=layer_type
                )
            )

        return added

    def _add_layers_with_plugins(
        self,
        path_or_paths: Union[str, Sequence[str]],
        kwargs: Optional[dict] = None,
        plugin: Optional[str] = None,
        layer_type: Optional[str] = None,
    ) -> List[layers.Layer]:
        """Load a path or a list of paths into the viewer using plugins.

        This function is mostly called from self.open_path, where the ``stack``
        argument determines whether a list of strings is handed to plugins one
        at a time, or en-masse.

        Parameters
        ----------
        path_or_paths : str or list of str
            A filepath, directory, or URL (or a list of any) to open. If a
            list, the assumption is that the list is to be treated as a stack.
        kwargs : dict, optional
            keyword arguments that will be used to overwrite any of those that
            are returned in the meta dict from plugins.
        plugin : str, optional
            Name of a plugin to use.  If provided, will force ``path`` to be
            read with the specified ``plugin``.  If the requested plugin cannot
            read ``path``, an exception will be raised.
        layer_type : str, optional
            If provided, will force data read from ``path`` to be passed to the
            corresponding ``add_<layer_type>`` method (along with any
            additional) ``kwargs`` provided to this function.  This *may*
            result in exceptions if the data returned from the path is not
            compatible with the layer_type.

        Returns
        -------
        List[layers.Layer]
            A list of any layers that were added to the viewer.
        """
        layer_data = read_data_with_plugins(path_or_paths, plugin=plugin)

        # glean layer names from filename. These will be used as *fallback*
        # names, if the plugin does not return a name kwarg in their meta dict.
        filenames = []
        if isinstance(path_or_paths, str):
            filenames = itertools.repeat(path_or_paths)
        elif is_sequence(path_or_paths):
            if len(path_or_paths) == len(layer_data):
                filenames = iter(path_or_paths)
            else:
                # if a list of paths has been returned as a list of layer data
                # without a 1:1 relationship between the two lists we iterate
                # over the first name
                filenames = itertools.repeat(path_or_paths[0])

        # add each layer to the viewer
        added: List[layers.Layer] = []  # for layers that get added
        for data, filename in zip(layer_data, filenames):
            basename, ext = os.path.splitext(os.path.basename(filename))
            _data = _unify_data_and_user_kwargs(
                data, kwargs, layer_type, fallback_name=basename
            )
            # actually add the layer
            new = self._add_layer_from_data(*_data)
            # some add_* methods return a List[Layer], others just a Layer
            # we want to always return a list
            added.extend(new if isinstance(new, list) else [new])
        return added

    def _add_layer_from_data(
        self, data, meta: dict = None, layer_type: Optional[str] = None
    ) -> Union[layers.Layer, List[layers.Layer]]:
        """Add arbitrary layer data to the viewer.

        Primarily intended for usage by reader plugin hooks.

        Parameters
        ----------
        data : Any
            Data in a format that is valid for the corresponding `add_*` method
            of the specified ``layer_type``.
        meta : dict, optional
            Dict of keyword arguments that will be passed to the corresponding
            `add_*` method.  MUST NOT contain any keyword arguments that are
            not valid for the corresponding method.
        layer_type : str
            Type of layer to add.  MUST have a corresponding add_* method on
            on the viewer instance.  If not provided, the layer is assumed to
            be "image", unless data.dtype is one of (np.int32, np.uint32,
            np.int64, np.uint64), in which case it is assumed to be "labels".

        Raises
        ------
        ValueError
            If ``layer_type`` is not one of the recognized layer types.
        TypeError
            If any keyword arguments in ``meta`` are unexpected for the
            corresponding `add_*` method for this layer_type.

        Examples
        --------
        A typical use case might be to upack a tuple of layer data with a
        specified layer_type.

        >>> viewer = napari.Viewer()
        >>> data = (
        ...     np.random.random((10, 2)) * 20,
        ...     {'face_color': 'blue'},
        ...     'points',
        ... )
        >>> viewer._add_layer_from_data(*data)

        """

        layer_type = (layer_type or '').lower()

        # assumes that big integer type arrays are likely labels.
        if not layer_type:
            layer_type = guess_labels(data)

        if layer_type not in layers.NAMES:
            raise ValueError(
                f"Unrecognized layer_type: '{layer_type}'. "
                f"Must be one of: {layers.NAMES}."
            )

        try:
            add_method = getattr(self, 'add_' + layer_type)
        except AttributeError:
            raise NotImplementedError(
                f"Sorry! {layer_type} is a valid layer type, but there is no "
                f"viewer.add_{layer_type} available yet."
            )

        try:
            layer = add_method(data, **(meta or {}))
        except TypeError as exc:
            if 'unexpected keyword argument' in str(exc):
                bad_key = str(exc).split('keyword argument ')[-1]
                raise TypeError(
                    "_add_layer_from_data received an unexpected keyword "
                    f"argument ({bad_key}) for layer type {layer_type}"
                ) from exc
            else:
                raise exc

        return layer


def _get_image_class() -> layers.Image:
    """Return Image or OctreeImage based config settings."""
    if config.async_octree:
        from ..layers.image.experimental.octree_image import OctreeImage

        return OctreeImage

    return layers.Image


def _create_remote_commands(layers: LayerList) -> None:
    """Start the monitor service if configured to use it."""
    if not config.monitor:
        return None

    from ..components.experimental.monitor import monitor
    from ..components.experimental.remote_commands import RemoteCommands

    monitor.start()  # Start if not already started.

    # Create a RemoteCommands object which will run commands
    # from remote clients that come through the monitor.
    return RemoteCommands(layers, monitor.run_command_event)


def _normalize_layer_data(data: LayerData) -> FullLayerData:
    """Accepts any layerdata tuple, and returns a fully qualified tuple.

    Parameters
    ----------
    data : LayerData
        1-, 2-, or 3-tuple with (data, meta, layer_type).

    Returns
    -------
    FullLayerData
        3-tuple with (data, meta, layer_type)

    Raises
    ------
    ValueError
        If data has len < 1 or len > 3, or if the second item in ``data`` is
        not a ``dict``, or the third item is not a valid layer_type ``str``
    """
    if not isinstance(data, tuple) and 0 < len(data) < 4:
        raise ValueError("LayerData must be a 1-, 2-, or 3-tuple")
    _data = list(data)
    if len(_data) > 1:
        if not isinstance(_data[1], dict):
            raise ValueError(
                "The second item in a LayerData tuple must be a dict"
            )
    else:
        _data.append(dict())
    if len(_data) > 2:
        if _data[2] not in layers.NAMES:
            raise ValueError(
                "The third item in a LayerData tuple must be one of: "
                f"{layers.NAMES!r}."
            )
    else:
        _data.append(guess_labels(_data[0]))
    return tuple(_data)  # type: ignore


def _unify_data_and_user_kwargs(
    data: LayerData,
    kwargs: Optional[dict] = None,
    layer_type: Optional[str] = None,
    fallback_name: str = None,
) -> FullLayerData:
    """Merge data returned from plugins with options specified by user.

    If ``data == (_data, _meta, _type)``.  Then:

    - ``kwargs`` will be used to update ``_meta``
    - ``layer_type`` will replace ``_type`` and, if provided, ``_meta`` keys
        will be pruned to layer_type-appropriate kwargs
    - ``fallback_name`` is used if ``not _meta.get('name')``

    .. note:

        If a user specified both layer_type and additional keyword arguments
        to viewer.open(), it is their responsibility to make sure the kwargs
        match the layer_type.

    Parameters
    ----------
    data : LayerData
        1-, 2-, or 3-tuple with (data, meta, layer_type) returned from plugin.
    kwargs : dict, optional
        User-supplied keyword arguments, to override those in ``meta`` supplied
        by plugins.
    layer_type : str, optional
        A user-supplied layer_type string, to override the ``layer_type``
        declared by the plugin.
    fallback_name : str, optional
        A name for the layer, to override any name in ``meta`` supplied by the
        plugin.

    Returns
    -------
    FullLayerData
        Fully qualified LayerData tuple with user-provided overrides.
    """
    _data, _meta, _type = _normalize_layer_data(data)

    if layer_type:
        # the user has explicitly requested this be a certain layer type
        # strip any kwargs from the plugin that are no longer relevant
        _meta = prune_kwargs(_meta, layer_type)
        _type = layer_type

    if kwargs:
        # if user provided kwargs, use to override any meta dict values that
        # were returned by the plugin. We only prune kwargs if the user did
        # *not* specify the layer_type. This means that if a user specified
        # both layer_type and additional keyword arguments to viewer.open(),
        # it is their responsibility to make sure the kwargs match the
        # layer_type.
        _meta.update(prune_kwargs(kwargs, _type) if not layer_type else kwargs)

    if not _meta.get('name') and fallback_name:
        _meta['name'] = fallback_name
    return (_data, _meta, _type)


def prune_kwargs(kwargs: Dict[str, Any], layer_type: str) -> Dict[str, Any]:
    """Return copy of ``kwargs`` with only keys valid for ``add_<layer_type>``

    Parameters
    ----------
    kwargs : dict
        A key: value mapping where some or all of the keys are parameter names
        for the corresponding ``Viewer.add_<layer_type>`` method.
    layer_type : str
        The type of layer that is going to be added with these ``kwargs``.

    Returns
    -------
    pruned_kwargs : dict
        A key: value mapping where all of the keys are valid parameter names
        for the corresponding ``Viewer.add_<layer_type>`` method.

    Raises
    ------
    ValueError
        If ``ViewerModel`` does not provide an ``add_<layer_type>`` method
        for the provided ``layer_type``.

    Examples
    --------
    >>> test_kwargs = {
    ...     'scale': (0.75, 1),
    ...     'blending': 'additive',
    ...     'num_colors': 10,
    ... }
    >>> prune_kwargs(test_kwargs, 'image')
    {'scale': (0.75, 1), 'blending': 'additive'}

    >>> # only labels has the ``num_colors`` argument
    >>> prune_kwargs(test_kwargs, 'labels')
    {'scale': (0.75, 1), 'blending': 'additive', 'num_colors': 10}
    """
    add_method = getattr(ViewerModel, 'add_' + layer_type, None)
    if not add_method or layer_type == 'layer':
        raise ValueError(f"Invalid layer_type: {layer_type}")

    # get valid params for the corresponding add_<layer_type> method
    valid = valid_add_kwargs()[layer_type]
    return {k: v for k, v in kwargs.items() if k in valid}


@lru_cache(maxsize=1)
def valid_add_kwargs() -> Dict[str, Set[str]]:
    """Return a dict where keys are layer types & values are valid kwargs."""
    valid = dict()
    for meth in dir(ViewerModel):
        if not meth.startswith('add_') or meth[4:] == 'layer':
            continue
        params = inspect.signature(getattr(ViewerModel, meth)).parameters
        valid[meth[4:]] = set(params) - {'self', 'kwargs'}
    return valid


for _layer in (
    layers.Labels,
    layers.Points,
    layers.Shapes,
    layers.Surface,
    layers.Tracks,
    layers.Vectors,
):
    func = create_add_method(_layer)
    setattr(ViewerModel, func.__name__, func)
