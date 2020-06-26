import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Optional
from copy import copy

import numpy as np

from ..layer_event_handler import LayerEventHandler
from ...components import Dims
from ...utils.dask_utils import configure_dask
from ...utils.event import EmitterGroup, Event
from ...utils.key_bindings import KeymapProvider
from ...utils.misc import ROOT_DIR
from ...utils.naming import magic_name
from ...utils.status_messages import format_float, status_format
from ..transforms import ScaleTranslate, TransformChain
from ..utils.layer_utils import compute_multiscale_level, convert_to_uint8
from ._base_constants import Blending


class Layer(KeymapProvider, ABC):
    """Base layer class.

    Parameters
    ----------
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.
    multiscale : bool
        Whether the data is multiscale or not. Multiscale data is
        represented by a list of data objects and should go from largest to
        smallest.

    Attributes
    ----------
    name : str
        Unique name of the layer.
    opacity : flaot
        Opacity of the layer visual, between 0.0 and 1.0.
    visible : bool
        Whether the layer visual is currently being displayed.
    blending : Blending
        Determines how RGB and alpha values get mixed.
            Blending.OPAQUE
                Allows for only the top layer to be visible and corresponds to
                depth_test=True, cull_face=False, blend=False.
            Blending.TRANSLUCENT
                Allows for multiple layers to be blended with different opacity
                and corresponds to depth_test=True, cull_face=False,
                blend=True, blend_func=('src_alpha', 'one_minus_src_alpha').
            Blending.ADDITIVE
                Allows for multiple layers to be blended together with
                different colors and opacity. Useful for creating overlays. It
                corresponds to depth_test=False, cull_face=False, blend=True,
                blend_func=('src_alpha', 'one').
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    multiscale : bool
        Whether the data is multiscale or not. Multiscale data is
        represented by a list of data objects and should go from largest to
        smallest.
    z_index : int
        Depth of the layer visual relative to other visuals in the scenecanvas.
    coordinates : tuple of float
        Coordinates of the cursor in the data space of each layer. The length
        of the tuple is equal to the number of dimensions of the layer.
    corner_pixels : array
        Coordinates of the top-left and bottom-right canvas pixels in the data
        space of each layer. The length of the tuple is equal to the number of
        dimensions of the layer.
    position : 2-tuple of int
        Cursor position in the image space of only the displayed dimensions.
    shape : tuple of int
        Size of the data in the layer.
    ndim : int
        Dimensionality of the layer.
    selected : bool
        Flag if layer is selected in the viewer or not.
    thumbnail : (N, M, 4) array
        Array of thumbnail data for the layer.
    status : str
        Displayed in status bar bottom left.
    help : str
        Displayed in status bar bottom right.
    interactive : bool
        Determine if canvas pan/zoom interactivity is enabled.
    cursor : str
        String identifying which cursor displayed over canvas.
    cursor_size : int | None
        Size of cursor if custom. None yields default size
    scale_factor : float
        Conversion factor from canvas coordinates to image coordinates, which
        depends on the current zoom level.


    Notes
    -----
    Must define the following:
        * `_get_range()`: called by `range` property
        * `data` property (setter & getter)

    May define the following:
        * `_set_view_slice(indices)`: called to set currently viewed slice
        * `_basename()`: base/default name of the layer
    """

    def __init__(
        self,
        data,
        ndim,
        *,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        opacity=1,
        blending='translucent',
        visible=True,
        multiscale=False,
    ):
        super().__init__()

        if name is None and data is not None:
            name = magic_name(data, path_prefix=ROOT_DIR)

        self.dask_optimized_slicing = configure_dask(data)
        self.metadata = metadata or {}
        self._opacity = opacity
        self._blending = Blending(blending)
        self._visible = visible
        self._selected = True
        self._freeze = False
        self._status = 'Ready'
        self._help = ''
        self._cursor = 'standard'
        self._cursor_size = None
        self._interactive = True
        self._value = None
        self.scale_factor = 1
        self.multiscale = multiscale

        self.dims = Dims(ndim)

        if scale is None:
            scale = [1] * ndim
        if translate is None:
            translate = [0] * ndim

        # Create a transform chain consisting of three transforms:
        # 1. `tile2data`: An initial transform only needed displaying tiles
        #   of an image. It maps pixels of the tile into the coordinate space
        #   of the full resolution data and can usually be represented by a
        #   scale factor and a translation. A common use case is viewing part
        #   of lower resolution level of a multiscale image, another is using a
        #   downsampled version of an image when the full image size is larger
        #   than the maximum allowed texture size of your graphics card.
        # 2. `data2world`: The main transform mapping data to a world-like
        #   coordinate.
        # 3. `world2grid`: An additional transform mapping world-coordinates
        #   into a grid for looking at layers side-by-side.
        self._transforms = TransformChain(
            [
                ScaleTranslate(
                    np.ones(ndim), np.zeros(ndim), name='tile2data'
                ),
                ScaleTranslate(scale, translate, name='data2world'),
                ScaleTranslate(
                    np.ones(ndim), np.zeros(ndim), name='world2grid'
                ),
            ]
        )

        self.coordinates = (0,) * ndim
        self._position = (0,) * self.dims.ndisplay
        self.corner_pixels = np.zeros((2, ndim), dtype=int)
        self._editable = True
        self._editable_history = True

        self._thumbnail_shape = (32, 32, 4)
        self._thumbnail = np.zeros(self._thumbnail_shape, dtype=np.uint8)
        self._update_properties = True
        if not name:
            self._name = self._basename()
        else:
            self._name = name

        self.event_handler = LayerEventHandler(component=self)

        self.events = EmitterGroup(
            source=self,
            auto_connect=True,
            refresh=Event,
            slice_data=Event,
            blending=Event,
            opacity=Event,
            visible=Event,
            selected=Event,
            scale=Event,
            translate=Event,
            data=Event,
            name=Event,
            name_unique=Event,
            thumbnail=Event,
            status=Event,
            help=Event,
            interactive=Event,
            cursor=Event,
            cursor_size=Event,
            editable=Event,
            event_handler_callback=self.event_handler.on_change,
        )

        self.dims.events.ndisplay.connect(lambda e: self._update_editable())
        self.dims.events.order.connect(self.refresh)
        self.dims.events.ndisplay.connect(self._update_dims)
        self.dims.events.order.connect(self._update_dims)
        self.dims.events.axis.connect(self.refresh)

        self.mouse_move_callbacks = []
        self.mouse_drag_callbacks = []
        self._persisted_mouse_event = {}
        self._mouse_drag_gen = {}

    def __str__(self):
        """Return self.name."""
        return self.name

    def __repr__(self):
        cls = type(self)
        return f"<{cls.__name__} layer {repr(self.name)} at {hex(id(self))}>"

    @classmethod
    def _basename(cls):
        return f'{cls.__name__}'

    @property
    def name(self):
        """str: Unique name of the layer."""
        return self._name

    @name.setter
    def name(self, value):
        if self.name == value:
            return
        old_name = copy(self.name)
        self.events.name_unique(value=(old_name, value))
        if self.name == old_name:
            self.events.name(value=value)

    def _on_name_change(self, value):
        self._name = value

    @property
    def opacity(self):
        """float: Opacity value between 0.0 and 1.0.
        """
        return self._opacity

    @opacity.setter
    def opacity(self, value):
        self.events.opacity(value=value)

    def _on_opacity_change(self, value):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                'Opacity must be between 0.0 and 1.0; ' f'got {value}.'
            )

        self._opacity = value
        self._update_thumbnail()
        self.status = format_float(self.opacity)

    @property
    def blending(self):
        """Blending mode: Determines how RGB and alpha values get mixed.

            Blending.OPAQUE
                Allows for only the top layer to be visible and corresponds to
                depth_test=True, cull_face=False, blend=False.
            Blending.TRANSLUCENT
                Allows for multiple layers to be blended with different opacity
                and corresponds to depth_test=True, cull_face=False,
                blend=True, blend_func=('src_alpha', 'one_minus_src_alpha').
            Blending.ADDITIVE
                Allows for multiple layers to be blended together with
                different colors and opacity. Useful for creating overlays. It
                corresponds to depth_test=False, cull_face=False, blend=True,
                blend_func=('src_alpha', 'one').
        """
        return str(self._blending)

    @blending.setter
    def blending(self, value):
        self.events.blending(value=value)

    def _on_blending_change(self, value):
        self._blending = Blending(value)

    @property
    def visible(self):
        """bool: Whether the visual is currently being displayed."""
        return self._visible

    @visible.setter
    def visible(self, value):
        self.events.visible(value=value)

    def _on_visible_change(self, value):
        self._visible = value
        self.refresh()
        if self.visible:
            self._update_editable()
        else:
            self.editable = False

    @property
    def editable(self):
        """bool: Whether the current layer data is editable from the viewer."""
        return self._editable

    @editable.setter
    def editable(self, value):
        self.events.editable(value=value)

    def _on_editable_change(self, value):
        self._editable = value

    def _update_editable(self):
        self.editable = self._is_editable

    @property
    def scale(self):
        """list: Anisotropy factors to scale data into world coordinates."""
        return self._transforms['data2world'].scale

    @scale.setter
    def scale(self, value):
        self.events.scale(value=np.array(value))

    def _on_scale_change(self, value):
        self._transforms['data2world'].scale = value
        self._update_dims()

    @property
    def translate(self):
        """list: Factors to shift the layer by in units of world coordinates."""
        return self._transforms['data2world'].translate

    @translate.setter
    def translate(self, value):
        self.events.translate(value=np.array(value))

    def _on_translate_change(self, value):
        self._transforms['data2world'].translate = np.array(value)
        self._update_dims()

    @property
    def translate_grid(self):
        """list: Factors to shift the layer by."""
        return self._transforms['world2grid'].translate

    @translate_grid.setter
    def translate_grid(self, value):
        if np.all(self.translate_grid == value):
            return
        self.events.translate(value=np.array(value))

    def _on_translate_grid_change(self, value):
        self._transforms['world2grid'].translate = value

    @property
    def position(self):
        """tuple of int: Cursor position in image of displayed dimensions."""
        return self._position

    @position.setter
    def position(self, value):
        if self._position == value:
            return
        self._position = value
        self._update_coordinates()

    def _update_dims(self, event=None):
        """Updates dims model, which is useful after data has been changed."""
        ndim = self._get_ndim()
        ndisplay = self.dims.ndisplay

        # If the dimensionality is changing then if the number of dimensions
        # is becoming smaller trim the property from the beginning, and if
        # the number of dimensions is becoming larger pad from the beginning
        if len(self.position) > ndisplay:
            self._position = self._position[-ndisplay:]
        elif len(self.position) < ndisplay:
            self._position = (0,) * (ndisplay - len(self.position)) + tuple(
                self.position
            )

        old_ndim = self.dims.ndim
        if old_ndim > ndim:
            keep_axes = range(old_ndim - ndim, old_ndim)
            self._transforms = self._transforms.set_slice(keep_axes)
        elif old_ndim < ndim:
            new_axes = range(ndim - old_ndim)
            self._transforms = self._transforms.expand_dims(new_axes)

        self.dims.ndim = ndim

        curr_range = self._get_range()
        for i, r in enumerate(curr_range):
            self.dims.set_range(i, r)

        self.refresh()
        self._update_coordinates()

    @property
    @abstractmethod
    def data(self):
        # user writes own docstring
        raise NotImplementedError()

    @data.setter
    @abstractmethod
    def data(self, value):
        raise NotImplementedError()

    @abstractmethod
    def _get_extent(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_ndim(self):
        raise NotImplementedError()

    @property
    def _is_editable(self):
        """Determine if editable based on layer properties."""
        return not self.dims.ndisplay == 3

    def _get_range(self):
        extent = self._get_extent()
        return tuple(
            (s * e[0], s * e[1], s) for e, s in zip(extent, self.scale)
        )

    def _get_base_state(self):
        """Get dictionary of attributes on base layer.

        Returns
        -------
        state : dict
            Dictionary of attributes on base layer.
        """
        base_dict = {
            'name': self.name,
            'metadata': self.metadata,
            'scale': list(self.scale),
            'translate': list(self.translate),
            'opacity': self.opacity,
            'blending': self.blending,
            'visible': self.visible,
        }
        return base_dict

    @abstractmethod
    def _get_state(self):
        raise NotImplementedError()

    @property
    def _type_string(self):
        return self.__class__.__name__.lower()

    def as_layer_data_tuple(self):
        state = self._get_state()
        state.pop('data', None)
        return self.data, state, self._type_string

    @property
    def thumbnail(self):
        """array: Integer array of thumbnail for the layer"""
        return self._thumbnail

    @thumbnail.setter
    def thumbnail(self, value):
        if 0 in value.shape:
            value = np.zeros(self._thumbnail_shape, dtype=np.uint8)
        if value.dtype != np.uint8:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = convert_to_uint8(value)

        padding_needed = np.subtract(self._thumbnail_shape, value.shape)
        pad_amounts = [(p // 2, (p + 1) // 2) for p in padding_needed]
        value = np.pad(value, pad_amounts, mode='constant')

        # blend thumbnail with opaque black background
        background = np.zeros(self._thumbnail_shape, dtype=np.uint8)
        background[..., 3] = 255

        f_dest = value[..., 3][..., None] / 255
        f_source = 1 - f_dest
        value = value * f_dest + background * f_source

        self.events.thumbnail(value=value.astype(np.uint8))

    def _on_thumbnail_change(self, value):
        self._thumbnail = value

    @property
    def ndim(self):
        """int: Number of dimensions in the data."""
        return self.dims.ndim

    @property
    def shape(self):
        """tuple of int: Shape of the data."""
        return tuple(
            np.round(r[1] - r[0]).astype(int) for r in self.dims.range
        )

    @property
    def selected(self):
        """bool: Whether this layer is selected or not."""
        return self._selected

    @selected.setter
    def selected(self, value):
        self.events.selected(value=value)

    def _on_selected_change(self, value):
        if value == self.selected:
            return
        self._selected = value

    @property
    def status(self):
        """str: displayed in status bar bottom left."""
        return self._status

    @status.setter
    def status(self, value):
        self.events.status(value=value)

    def _on_status_change(self, value):
        self._status = value

    @property
    def help(self):
        """str: displayed in status bar bottom right."""
        return self._help

    @help.setter
    def help(self, value):
        self.events.help(value=value)

    def _on_help_change(self, value):
        self._help = value

    @property
    def interactive(self):
        """bool: Determine if canvas pan/zoom interactivity is enabled."""
        return self._interactive

    @interactive.setter
    def interactive(self, value):
        self.events.interactive(value=value)

    def _on_interactive_change(self, value):
        self._interactive = value

    @property
    def cursor(self):
        """str: String identifying cursor displayed over canvas."""
        return self._cursor

    @cursor.setter
    def cursor(self, value):
        self.events.cursor(value=value)

    def _on_cursor_change(self, value):
        self._cursor = value

    @property
    def cursor_size(self):
        """int | None: Size of cursor if custom. None yields default size."""
        return self._cursor_size

    @cursor_size.setter
    def cursor_size(self, value):
        self.events.cursor_size(value=value)

    def _on_cursor_size_change(self, value):
        self._cursor_size = value

    def set_view_slice(self):
        with self.dask_optimized_slicing():
            self._set_view_slice()

    @abstractmethod
    def _set_view_slice(self):
        raise NotImplementedError()

    @abstractmethod
    def _update_thumbnail(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_value(self):
        raise NotImplementedError()

    def get_value(self):
        """Value of data at current coordinates.

        Returns
        -------
        value : tuple, None
            Value of the data at the coordinates.
        """
        if self.visible:
            return self._get_value()
        else:
            return None

    @contextmanager
    def block_update_properties(self):
        self._update_properties = False
        yield
        self._update_properties = True

    def _set_highlight(self, force=False):
        """Render layer highlights when appropriate.

        Parameters
        ----------
        force : bool
            Bool that forces a redraw to occur when `True`.
        """
        pass

    def refresh(self, event=None):
        """Refresh all layer data based on current view slice.
        """
        if self.visible:
            self.set_view_slice()
            self.events.slice_data(value=None)
            self._update_thumbnail()
            self._update_coordinates()
            self._set_highlight(force=True)

    def _update_coordinates(self):
        """Insert the cursor position into the correct position in the
        tuple of indices and update the cursor coordinates.
        """
        coords = list(self.dims.indices)
        for d, p in zip(self.dims.displayed, self.position):
            coords[d] = p
        self.coordinates = tuple(coords)
        self._value = self.get_value()
        self.status = self.get_message()

    def _update_multiscale(self, corner_pixels, shape_threshold):
        """Refresh layer multiscale if new resolution level or tile is required.

        Parameters
        ----------
        corner_pixels : array
            Coordinates of the top-left and bottom-right canvas pixels in the
            data space of each layer. The length of the tuple is equal to the
            number of dimensions of the layer. If different from the current
            layer corner_pixels the layer needs refreshing.
        shape_threshold : tuple
            Requested shape of field of view in data coordinates
        """

        if len(self.dims.displayed) == 3:
            data_level = corner_pixels.shape[1] - 1
        else:
            # Clip corner pixels inside data shape
            new_corner_pixels = np.clip(
                self.corner_pixels,
                0,
                np.subtract(self.level_shapes[self.data_level], 1),
            )

            # Scale to full resolution of the data
            requested_shape = (
                new_corner_pixels[1] - new_corner_pixels[0]
            ) * self.downsample_factors[self.data_level]

            downsample_factors = self.downsample_factors[
                :, self.dims.displayed
            ]

            data_level = compute_multiscale_level(
                requested_shape[self.dims.displayed],
                shape_threshold,
                downsample_factors,
            )

        if data_level != self.data_level:
            # Set the data level, which will trigger a layer refresh and
            # further updates including recalculation of the corner_pixels
            # for the new level
            self.data_level = data_level
            self.refresh()
        elif not np.all(self.corner_pixels == corner_pixels):
            self.refresh()

    @property
    def displayed_coordinates(self):
        """list: List of currently displayed coordinates."""
        return [self.coordinates[i] for i in self.dims.displayed]

    def get_message(self):
        """Generate a status message based on the coordinates and value

        Returns
        ----------
        msg : string
            String containing a message that can be used as a status update.
        """
        coordinates = self._transforms.simplified(self.coordinates)
        full_coord = np.round(coordinates).astype(int)

        msg = f'{self.name} {full_coord}'

        value = self._value
        if value is not None:
            if isinstance(value, tuple) and value != (None, None):
                # it's a multiscale -> value = (data_level, value)
                msg += f': {status_format(value[0])}'
                if value[1] is not None:
                    msg += f', {status_format(value[1])}'
            else:
                # it's either a grayscale or rgb image (scalar or list)
                msg += f': {status_format(value)}'
        return msg

    def save(self, path: str, plugin: Optional[str] = None) -> List[str]:
        """Save this layer to ``path`` with default (or specified) plugin.

        Parameters
        ----------
        path : str
            A filepath, directory, or URL to open.  Extensions may be used to
            specify output format (provided a plugin is avaiable for the
            requested format).
        plugin : str, optional
            Name of the plugin to use for saving. If ``None`` then all plugins
            corresponding to appropriate hook specification will be looped
            through to find the first one that can save the data.

        Returns
        -------
        list of str
            File paths of any files that were written.
        """
        from ...plugins.io import save_layers

        return save_layers(path, [self], plugin=plugin)
