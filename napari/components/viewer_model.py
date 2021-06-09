from __future__ import annotations

import inspect
import itertools
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from pydantic import Extra, Field, validator

from .. import layers
from ..layers import Image, Layer
from ..layers._source import layer_source
from ..layers.image._image_utils import guess_labels
from ..layers.utils.stack_utils import split_channels
from ..types import PathOrPaths
from ..utils._register import create_func as create_add_method
from ..utils.colormaps import ensure_colormap
from ..utils.events import Event, EventedModel, disconnect_events
from ..utils.events.event import WarningEmitter
from ..utils.key_bindings import KeymapProvider
from ..utils.misc import is_sequence
from ..utils.mouse_bindings import MousemapProvider
from ..utils.settings import SETTINGS
from ..utils.theme import available_themes
from ..utils.translations import trans
from ._viewer_mouse_bindings import dims_scroll
from .axes import Axes
from .camera import Camera
from .cursor import Cursor
from .dims import Dims
from .grid import GridCanvas
from .layerlist import LayerList
from .scale_bar import ScaleBar
from .text_overlay import TextOverlay
from .tooltip import Tooltip

DEFAULT_THEME = 'dark'
EXCLUDE_DICT = {
    'keymap',
    '_mouse_wheel_gen',
    '_mouse_drag_gen',
    '_persisted_mouse_event',
    'mouse_move_callbacks',
    'mouse_drag_callbacks',
    'mouse_wheel_callbacks',
}
EXCLUDE_JSON = EXCLUDE_DICT.union({'layers', 'active_layer'})

if TYPE_CHECKING:
    from ..types import FullLayerData, LayerData


__all__ = ['ViewerModel', 'valid_add_kwargs']


# KeymapProvider & MousemapProvider should eventually be moved off the ViewerModel
class ViewerModel(KeymapProvider, MousemapProvider, EventedModel):
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
    axis_labels : list of str
        Dimension names.

    Attributes
    ----------
    window : Window
        Parent window.
    layers : LayerList
        List of contained layers.
    dims : Dimensions
        Contains axes, indices, dimensions and sliders.
    """

    # Using allow_mutation=False means these attributes aren't settable and don't
    # have an event emitter associated with them
    axes: Axes = Field(default_factory=Axes, allow_mutation=False)
    camera: Camera = Field(default_factory=Camera, allow_mutation=False)
    cursor: Cursor = Field(default_factory=Cursor, allow_mutation=False)
    dims: Dims = Field(default_factory=Dims, allow_mutation=False)
    grid: GridCanvas = Field(default_factory=GridCanvas, allow_mutation=False)
    layers: LayerList = Field(
        default_factory=LayerList, allow_mutation=False
    )  # Need to create custom JSON encoder for layer!
    scale_bar: ScaleBar = Field(default_factory=ScaleBar, allow_mutation=False)
    text_overlay: TextOverlay = Field(
        default_factory=TextOverlay, allow_mutation=False
    )

    help: str = ''
    status: str = 'Ready'
    tooltip: Tooltip = Field(default_factory=Tooltip, allow_mutation=False)
    theme: str = DEFAULT_THEME
    title: str = 'napari'

    # 2-tuple indicating height and width
    _canvas_size: Tuple[int, int] = (600, 800)

    def __init__(self, title='napari', ndisplay=2, order=(), axis_labels=()):
        # allow extra attributes during model initialization, useful for mixins
        self.__config__.extra = Extra.allow
        super().__init__(
            title=title,
            dims={
                'axis_labels': axis_labels,
                'ndisplay': ndisplay,
                'order': order,
            },
        )
        self.__config__.extra = Extra.ignore
        self.tooltip.visible = SETTINGS.appearance.layer_tooltip_visibility
        SETTINGS.appearance.events.layer_tooltip_visibility.connect(
            self._tooltip_visible_update
        )

        # Add extra events - ideally these will be removed too!
        self.events.add(layers_change=Event, reset_view=Event)

        # Connect events
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
        self.layers.selection.events.active.connect(self._on_active_layer)

        # Add mouse callback
        self.mouse_wheel_callbacks.append(dims_scroll)

        self.events.add(
            # FIXME: Deferred translation?
            active_layer=WarningEmitter(
                trans._(
                    "'viewer.events.active_layer' is deprecated and will be removed in napari v0.4.9, use 'viewer.layers.selection.events.active' instead",
                    deferred=True,
                ),
                type='active_layer',
            )
        )

    def _tooltip_visible_update(self, event):
        self.tooltip.visible = event.value

    @validator('theme')
    def _valid_theme(cls, v):
        themes = available_themes()
        if v not in available_themes():
            raise ValueError(
                trans._(
                    "Theme '{theme_name}' not found; options are {themes}.",
                    deferred=True,
                    theme_name=v,
                    themes=themes,
                )
            )

        return v

    def json(self, **kwargs):
        """Serialize to json."""
        # Manually exclude the layer list and active layer which cannot be serialized at this point
        # and mouse and keybindings don't belong on model
        # https://github.com/samuelcolvin/pydantic/pull/2231
        # https://github.com/samuelcolvin/pydantic/issues/660#issuecomment-642211017
        exclude = kwargs.pop('exclude', set())
        exclude = exclude.union(EXCLUDE_JSON)
        return super().json(exclude=exclude, **kwargs)

    def dict(self, **kwargs):
        """Convert to a dictionaty."""
        # Manually exclude the layer list and active layer which cannot be serialized at this point
        # and mouse and keybindings don't belong on model
        # https://github.com/samuelcolvin/pydantic/pull/2231
        # https://github.com/samuelcolvin/pydantic/issues/660#issuecomment-642211017
        exclude = kwargs.pop('exclude', set())
        exclude = exclude.union(EXCLUDE_DICT)
        return super().dict(exclude=exclude, **kwargs)

    def __hash__(self):
        return id(self)

    def __str__(self):
        """Simple string representation"""
        return f'napari.Viewer: {self.title}'

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
            self.camera.zoom = 0.95 * np.min(
                np.array(self._canvas_size) / np.array(size[-2:])
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

    def _on_active_layer(self, event):
        """Update viewer state for a new active layer."""
        active_layer = event.value
        if active_layer is None:
            self.help = ''
            self.cursor.style = 'standard'
            self.camera.interactive = True
        else:
            self.help = active_layer.help
            self.cursor.style = active_layer.cursor
            self.cursor.size = active_layer.cursor_size
            self.camera.interactive = active_layer.interactive

    @property
    def active_layer(self):
        warnings.warn(
            trans._(
                "'viewer.active_layer' is deprecated and will be removed in napari v0.4.9.  Please use 'viewer.layers.selection.active' instead.",
                deferred=True,
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self.layers.selection.active

    def __setattr__(self, name: str, value: Any) -> None:
        # this method is only for the deprecation warning, because pydantic
        # prevents using @active_layer.setter
        if name != 'active_layer':
            return super().__setattr__(name, value)

        warnings.warn(
            trans._(
                "'viewer.active_layer' is deprecated and will be removed in napari v0.4.9.  Please use 'viewer.layers.selection.active' instead.",
                deferred=True,
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        self.layers.selection.active = value

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
        self.cursor.position = (0,) * self.dims.ndim
        self.events.layers_change()

    def _update_interactive(self, event):
        """Set the viewer interactivity with the `event.interactive` bool."""
        self.camera.interactive = event.interactive

    def _update_cursor(self, event):
        """Set the viewer cursor with the `event.cursor` string."""
        self.cursor.style = event.cursor

    def _update_cursor_size(self, event):
        """Set the viewer cursor_size with the `event.cursor_size` int."""
        self.cursor.size = event.cursor_size

    def _on_cursor_position_change(self, event):
        """Set the layer cursor position."""
        with warnings.catch_warnings():
            # Catch the deprecation warning on layer.position
            warnings.filterwarnings(
                'ignore',
                message=str(
                    trans._('layer.position is deprecated', deferred=True)
                ),
            )
            for layer in self.layers:
                layer.position = self.cursor.position

        # Update status and help bar based on active layer
        active = self.layers.selection.active
        if active is not None:
            self.status = active.get_status(self.cursor.position, world=True)
            self.help = active.help
            if self.tooltip.visible:
                self.tooltip.text = active._get_tooltip_text(
                    self.cursor.position, world=True
                )

    def _on_grid_change(self, event):
        """Arrange the current layers is a 2D grid."""
        extent = self._sliced_extent_world
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            i_row, i_column = self.grid.position(n_layers - 1 - i, n_layers)
            self._subplot(layer, (i_row, i_column), extent)

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

        # Connect individual layer events to viewer events
        # TODO: in a future PR, we should now be able to connect viewer *only*
        # to viewer.layers.events... and avoid direct viewer->layer connections
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
        event : napari.utils.event.Event
            Event which will remove a layer.

        Returns
        -------
        layer : :class:`napari.layers.Layer` or list
            The layer that was added (same as input).
        """
        layer = event.value

        # Disconnect all connections from layer
        disconnect_events(layer.events, self)
        disconnect_events(layer.events, self.layers)

        self._on_layers_change(None)
        self._on_grid_change(None)

    def add_layer(self, layer: Layer) -> Layer:
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
    ) -> Union[Image, List[Image]]:
        """Add an image layer to the layer list.

        Parameters
        ----------
        data : array or list of array
            Image data. Can be N >= 2 dimensional. If the last dimension has length
            3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a
            list and arrays are decreasing in shape then the data is treated as
            a multiscale image. Please note multiscale rendering is only
            supported in 2D. In 3D, only the lowest resolution scale is
            displayed.
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
        affine : n-D array or napari.utils.transforms.Affine
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
            should be the largest. Please note multiscale rendering is only
            supported in 2D. In 3D, only the lowest resolution scale is
            displayed.

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

        if channel_axis is None:
            kwargs['colormap'] = kwargs['colormap'] or 'gray'
            kwargs['blending'] = kwargs['blending'] or 'translucent'
            # Helpful message if someone tries to add mulit-channel kwargs,
            # but forget the channel_axis arg
            for k, v in kwargs.items():
                if k not in iterable_kwargs and is_sequence(v):
                    raise TypeError(
                        trans._(
                            "Received sequence for argument '{argument}', did you mean to specify a 'channel_axis'? ",
                            deferred=True,
                            argument=k,
                        )
                    )
            layer = Image(data, **kwargs)
            self.layers.append(layer)

            return layer
        else:
            layerdata_list = split_channels(data, channel_axis, **kwargs)

            layer_list = list()
            for image, i_kwargs, _ in layerdata_list:
                layer = Image(image, **i_kwargs)
                self.layers.append(layer)
                layer_list.append(layer)

            return layer_list

    def open_sample(
        self,
        plugin: str,
        sample: str,
        reader_plugin: Optional[str] = None,
        **kwargs,
    ) -> List[Layer]:
        """Open `sample` from `plugin` and add it to the viewer.

        To see all available samples registered by plugins, use
        :func:`napari.plugins.available_samples`

        Parameters
        ----------
        plugin : str
            name of a plugin providing a sample
        sample : str
            name of the sample
        reader_plugin : str, optional
            reader plugin to pass to viewer.open (only used if the sample data
            is a string).  by default None.
        **kwargs
            additional kwargs will be passed to the sample data loader provided
            by `plugin`.  Use of **kwargs may raise an error if the kwargs do
            not match the sample data loader.

        Returns
        -------
        layers : list
            A list of any layers that were added to the viewer.

        Raises
        ------
        KeyError
            If `plugin` does not provide a sample named `sample`.
        """
        from ..plugins import plugin_manager

        try:
            data = plugin_manager._sample_data[plugin][sample]['data']
        except KeyError:
            samples = plugin_manager.available_samples()
            msg = trans._(
                "Plugin {plugin!r} does not provide sample data named {sample!r}. ",
                plugin=plugin,
                sample=sample,
                deferred=True,
            )
            if samples:
                msg = trans._(
                    "Plugin {plugin!r} does not provide sample data named {sample!r}. Available samples include: {samples}.",
                    deferred=True,
                    plugin=plugin,
                    sample=sample,
                    samples=samples,
                )
            else:
                msg = trans._(
                    "Plugin {plugin!r} does not provide sample data named {sample!r}. No plugin samples have been registered.",
                    deferred=True,
                    plugin=plugin,
                    sample=sample,
                )

            raise KeyError(msg)

        with layer_source(sample=(plugin, sample)):
            if callable(data):
                added = []
                for datum in data(**kwargs):
                    added.extend(self._add_layer_from_data(*datum))
                return added
            elif isinstance(data, (str, Path)):
                return self.open(data, plugin=reader_plugin)
            else:
                raise TypeError(
                    trans._(
                        'Got unexpected type for sample ({plugin!r}, {sample!r}): {data_type}',
                        deferred=True,
                        plugin=plugin,
                        sample=sample,
                        data_type=type(data),
                    )
                )

    def open(
        self,
        path: PathOrPaths,
        *,
        stack: bool = False,
        plugin: Optional[str] = None,
        layer_type: Optional[str] = None,
        **kwargs,
    ) -> List[Layer]:
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
        paths = [path] if isinstance(path, (Path, str)) else path
        paths = [os.fspath(path) for path in paths]  # PathObjects -> str
        if not isinstance(paths, (tuple, list)):
            raise ValueError(
                trans._(
                    "'path' argument must be a string, list, or tuple",
                    deferred=True,
                )
            )

        if stack:
            return self._add_layers_with_plugins(
                paths, kwargs, plugin=plugin, layer_type=layer_type
            )

        added: List[Layer] = []  # for layers that get added
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
    ) -> List[Layer]:
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
        List[Layer]
            A list of any layers that were added to the viewer.
        """
        from ..plugins.io import read_data_with_plugins

        layer_data, hookimpl = read_data_with_plugins(
            path_or_paths, plugin=plugin
        )

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
        added: List[Layer] = []  # for layers that get added
        plugin = hookimpl.plugin_name if hookimpl else None
        for data, filename in zip(layer_data, filenames):
            basename, _ext = os.path.splitext(os.path.basename(filename))
            _data = _unify_data_and_user_kwargs(
                data, kwargs, layer_type, fallback_name=basename
            )
            # actually add the layer
            with layer_source(path=filename, reader_plugin=plugin):
                added.extend(self._add_layer_from_data(*_data))
        return added

    def _add_layer_from_data(
        self,
        data,
        meta: Dict[str, Any] = None,
        layer_type: Optional[str] = None,
    ) -> List[Layer]:
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

        Returns
        -------
        layers : list of layers
            A list of layers added to the viewer.

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
                trans._(
                    "Unrecognized layer_type: '{layer_type}'. Must be one of: {layer_names}.",
                    deferred=True,
                    layer_type=layer_type,
                    layer_names=layers.NAMES,
                )
            )

        try:
            add_method = getattr(self, 'add_' + layer_type)
            layer = add_method(data, **(meta or {}))
        except TypeError as exc:
            if 'unexpected keyword argument' not in str(exc):
                raise exc
            bad_key = str(exc).split('keyword argument ')[-1]
            raise TypeError(
                trans._(
                    "_add_layer_from_data received an unexpected keyword argument ({bad_key}) for layer type {layer_type}",
                    deferred=True,
                    bad_key=bad_key,
                    layer_type=layer_type,
                )
            ) from exc
        return layer if isinstance(layer, list) else [layer]


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
        raise ValueError(
            trans._(
                "LayerData must be a 1-, 2-, or 3-tuple",
                deferred=True,
            )
        )

    _data = list(data)
    if len(_data) > 1:
        if not isinstance(_data[1], dict):
            raise ValueError(
                trans._(
                    "The second item in a LayerData tuple must be a dict",
                    deferred=True,
                )
            )
    else:
        _data.append(dict())
    if len(_data) > 2:
        if _data[2] not in layers.NAMES:
            raise ValueError(
                trans._(
                    "The third item in a LayerData tuple must be one of: {layers!r}.",
                    deferred=True,
                    layers=layers.NAMES,
                )
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
        raise ValueError(
            trans._(
                "Invalid layer_type: {layer_type}",
                deferred=True,
                layer_type=layer_type,
            )
        )

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
