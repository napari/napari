from __future__ import annotations

import itertools
import os.path
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import cached_property
from typing import List, Optional, Tuple, Union

import magicgui as mgui
import numpy as np
import numpy.typing as npt
from npe2 import plugin_manager as pm

from napari.layers.base._base_constants import Blending, Mode
from napari.layers.base._base_mouse_bindings import (
    highlight_box_handles,
    transform_with_box,
)
from napari.layers.utils._slice_input import _SliceInput
from napari.layers.utils.interactivity_utils import (
    drag_data_to_projected_distance,
)
from napari.layers.utils.layer_utils import (
    coerce_affine,
    compute_multiscale_level_and_corners,
    convert_to_uint8,
    dims_displayed_world_to_layer,
    get_extent_world,
)
from napari.layers.utils.plane import ClippingPlane, ClippingPlaneList
from napari.utils._dask_utils import configure_dask
from napari.utils._magicgui import (
    add_layer_to_viewer,
    add_layers_to_viewer,
    get_layers,
)
from napari.utils.events import EmitterGroup, Event, EventedDict
from napari.utils.events.event import WarningEmitter
from napari.utils.geometry import (
    find_front_back_face,
    intersect_line_with_axis_aligned_bounding_box_3d,
)
from napari.utils.key_bindings import KeymapProvider
from napari.utils.mouse_bindings import MousemapProvider
from napari.utils.naming import magic_name
from napari.utils.status_messages import generate_layer_coords_status
from napari.utils.transforms import Affine, CompositeAffine, TransformChain
from napari.utils.translations import trans

Extent = namedtuple('Extent', 'data world step')


def no_op(layer: Layer, event: Event) -> None:
    """
    A convenient no-op event for the layer mouse binding.

    This makes it easier to handle many cases by inserting this as
    as place holder

    Parameters
    ----------
    layer : Layer
        Current layer on which this will be bound as a callback
    event : Event
        event that triggered this mouse callback.

    Returns
    -------
    None

    """
    return


@mgui.register_type(choices=get_layers, return_callback=add_layer_to_viewer)
class Layer(KeymapProvider, MousemapProvider, ABC):
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
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', 'translucent_no_depth', 'additive', and 'minimum'}.
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
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    visible : bool
        Whether the layer visual is currently being displayed.
    blending : Blending
        Determines how RGB and alpha values get mixed.

        * ``Blending.OPAQUE``
          Allows for only the top layer to be visible and corresponds to
          ``depth_test=True``, ``cull_face=False``, ``blend=False``.
        * ``Blending.TRANSLUCENT``
          Allows for multiple layers to be blended with different opacity and
          corresponds to ``depth_test=True``, ``cull_face=False``,
          ``blend=True``, ``blend_func=('src_alpha', 'one_minus_src_alpha')``,
          and ``blend_equation=('func_add')``.
        * ``Blending.TRANSLUCENT_NO_DEPTH``
          Allows for multiple layers to be blended with different opacity, but
          no depth testing is performed. Corresponds to ``depth_test=False``,
          ``cull_face=False``, ``blend=True``,
          ``blend_func=('src_alpha', 'one_minus_src_alpha')``, and
          ``blend_equation=('func_add')``.
        * ``Blending.ADDITIVE``
          Allows for multiple layers to be blended together with different
          colors and opacity. Useful for creating overlays. It corresponds to
          ``depth_test=False``, ``cull_face=False``, ``blend=True``,
          ``blend_func=('src_alpha', 'one')``, and ``blend_equation=('func_add')``.
        * ``Blending.MINIMUM``
            Allows for multiple layers to be blended together such that
            the minimum of each RGB component and alpha are selected.
            Useful for creating overlays with inverted colormaps. It
            corresponds to ``depth_test=False``, ``cull_face=False``, ``blend=True``,
            ``blend_equation=('min')``.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    multiscale : bool
        Whether the data is multiscale or not. Multiscale data is
        represented by a list of data objects and should go from largest to
        smallest.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    z_index : int
        Depth of the layer visual relative to other visuals in the scenecanvas.
    corner_pixels : array
        Coordinates of the top-left and bottom-right canvas pixels in the data
        coordinates of each layer. For multiscale data the coordinates are in
        the space of the currently viewed data level, not the highest resolution
        level.
    ndim : int
        Dimensionality of the layer.
    thumbnail : (N, M, 4) array
        Array of thumbnail data for the layer.
    status : str
        Displayed in status bar bottom left.
    help : str
        Displayed in status bar bottom right.
    interactive : bool
        Determine if canvas pan/zoom interactivity is enabled.
        This attribute is deprecated since 0.5.0 and should not be used.
        Use the mouse_pan and mouse_zoom attributes instead.
    mouse_pan : bool
        Determine if canvas interactive panning is enabled with the mouse.
    mouse_zoom : bool
        Determine if canvas interactive zooming is enabled with the mouse.
    cursor : str
        String identifying which cursor displayed over canvas.
    cursor_size : int | None
        Size of cursor if custom. None yields default size
    scale_factor : float
        Conversion factor from canvas coordinates to image coordinates, which
        depends on the current zoom level.
    source : Source
        source of the layer (such as a plugin or widget)

    Notes
    -----
    Must define the following:

    * `_extent_data`: property
    * `data` property (setter & getter)

    May define the following:

    * `_set_view_slice()`: called to set currently viewed slice
    * `_basename()`: base/default name of the layer
    """

    _modeclass = Mode

    _drag_modes = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: transform_with_box,
    }

    _move_modes = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: highlight_box_handles,
    }
    _cursor_modes = {
        Mode.PAN_ZOOM: 'standard',
        Mode.TRANSFORM: 'standard',
    }

    def __init__(
        self,
        data,
        ndim,
        *,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=1,
        blending='translucent',
        visible=True,
        multiscale=False,
        cache=True,  # this should move to future "data source" object.
        experimental_clipping_planes=None,
        mode='pan_zoom',
    ) -> None:
        super().__init__()

        if name is None and data is not None:
            name = magic_name(data)

        if scale is not None and not np.all(scale):
            raise ValueError(
                trans._(
                    "Layer {name} is invalid because it has scale values of 0. The layer's scale is currently {scale}",
                    deferred=True,
                    name=repr(name),
                    scale=repr(scale),
                )
            )

        # Needs to be imported here to avoid circular import in _source
        from napari.layers._source import current_source

        self._source = current_source()
        self.dask_optimized_slicing = configure_dask(data, cache)
        self._metadata = dict(metadata or {})
        self._opacity = opacity
        self._blending = Blending(blending)
        self._visible = visible
        self._freeze = False
        self._status = 'Ready'
        self._help = ''
        self._cursor = 'standard'
        self._cursor_size = 1
        self._mouse_pan = True
        self._mouse_zoom = True
        self._value = None
        self.scale_factor = 1
        self.multiscale = multiscale
        self._experimental_clipping_planes = ClippingPlaneList()
        self._mode = self._modeclass('pan_zoom')

        self._ndim = ndim

        self._slice_input = _SliceInput(
            ndisplay=2,
            point=(0,) * ndim,
            order=tuple(range(ndim)),
        )

        # Create a transform chain consisting of four transforms:
        # 1. `tile2data`: An initial transform only needed to display tiles
        #   of an image. It maps pixels of the tile into the coordinate space
        #   of the full resolution data and can usually be represented by a
        #   scale factor and a translation. A common use case is viewing part
        #   of lower resolution level of a multiscale image, another is using a
        #   downsampled version of an image when the full image size is larger
        #   than the maximum allowed texture size of your graphics card.
        # 2. `data2physical`: The main transform mapping data to a world-like
        #   physical coordinate that may also encode acquisition parameters or
        #   sample spacing.
        # 3. `physical2world`: An extra transform applied in world-coordinates that
        #   typically aligns this layer with another.
        # 4. `world2grid`: An additional transform mapping world-coordinates
        #   into a grid for looking at layers side-by-side.
        if scale is None:
            scale = [1] * ndim
        if translate is None:
            translate = [0] * ndim
        self._transforms = TransformChain(
            [
                Affine(np.ones(ndim), np.zeros(ndim), name='tile2data'),
                CompositeAffine(
                    scale,
                    translate,
                    rotate=rotate,
                    shear=shear,
                    ndim=ndim,
                    name='data2physical',
                ),
                coerce_affine(affine, ndim=ndim, name='physical2world'),
                Affine(np.ones(ndim), np.zeros(ndim), name='world2grid'),
            ]
        )

        self.corner_pixels = np.zeros((2, ndim), dtype=int)
        self._editable = True
        self._array_like = False

        self._thumbnail_shape = (32, 32, 4)
        self._thumbnail = np.zeros(self._thumbnail_shape, dtype=np.uint8)
        self._update_properties = True
        self._name = ''
        self.experimental_clipping_planes = experimental_clipping_planes

        # circular import
        from napari.components.overlays.bounding_box import BoundingBoxOverlay
        from napari.components.overlays.interaction_box import (
            SelectionBoxOverlay,
            TransformBoxOverlay,
        )

        self._overlays = EventedDict()

        self.events = EmitterGroup(
            source=self,
            refresh=Event,
            set_data=Event,
            blending=Event,
            opacity=Event,
            visible=Event,
            scale=Event,
            translate=Event,
            rotate=Event,
            shear=Event,
            affine=Event,
            data=Event,
            name=Event,
            thumbnail=Event,
            status=Event,
            help=Event,
            interactive=WarningEmitter(
                trans._(
                    "layer.events.interactive is deprecated since 0.4.18 and will be removed in 0.6.0. Please use layer.events.mouse_pan and layer.events.mouse_zoom",
                    deferred=True,
                ),
                type_name='interactive',
            ),
            mouse_pan=Event,
            mouse_zoom=Event,
            cursor=Event,
            cursor_size=Event,
            editable=Event,
            loaded=Event,
            extent=Event,
            _overlays=Event,
            select=WarningEmitter(
                trans._(
                    "'layer.events.select' is deprecated and will be removed in napari v0.4.9, use 'viewer.layers.selection.events.changed' instead, and inspect the 'added' attribute on the event.",
                    deferred=True,
                ),
                type_name='select',
            ),
            deselect=WarningEmitter(
                trans._(
                    "'layer.events.deselect' is deprecated and will be removed in napari v0.4.9, use 'viewer.layers.selection.events.changed' instead, and inspect the 'removed' attribute on the event.",
                    deferred=True,
                ),
                type_name='deselect',
            ),
            mode=Event,
        )
        self.name = name
        self.mode = mode
        self._overlays.update(
            {
                'transform_box': TransformBoxOverlay(),
                'selection_box': SelectionBoxOverlay(),
                'bounding_box': BoundingBoxOverlay(),
            }
        )

        # TODO: we try to avoid inner event connection, but this might be the only way
        #       until we figure out nested evented objects
        self._overlays.events.connect(self.events._overlays)

    def __str__(self):
        """Return self.name."""
        return self.name

    def __repr__(self):
        cls = type(self)
        return f"<{cls.__name__} layer {self.name!r} at {hex(id(self))}>"

    def _mode_setter_helper(self, mode):
        """
        Helper to manage callbacks in multiple layers

        Parameters
        ----------
        mode : type(self._modeclass) | str
            New mode for the current layer.

        Returns
        -------
        bool : whether mode changed

        """
        mode = self._modeclass(mode)
        assert mode is not None
        if not self.editable:
            mode = self._modeclass.PAN_ZOOM
        if mode == self._mode:
            return mode

        if mode.value not in self._modeclass.keys():  # noqa: SIM118
            raise ValueError(
                trans._(
                    "Mode not recognized: {mode}", deferred=True, mode=mode
                )
            )

        for callback_list, mode_dict in [
            (self.mouse_drag_callbacks, self._drag_modes),
            (self.mouse_move_callbacks, self._move_modes),
            (
                self.mouse_double_click_callbacks,
                getattr(
                    self, '_double_click_modes', defaultdict(lambda: no_op)
                ),
            ),
        ]:
            if mode_dict[self._mode] in callback_list:
                callback_list.remove(mode_dict[self._mode])
            callback_list.append(mode_dict[mode])
        self.cursor = self._cursor_modes[mode]

        self.mouse_pan = mode == self._modeclass.PAN_ZOOM
        self._overlays['transform_box'].visible = (
            mode == self._modeclass.TRANSFORM
        )

        if mode == self._modeclass.TRANSFORM:
            self.help = trans._(
                'hold <space> to pan/zoom, hold <shift> to preserve aspect ratio and rotate in 45° increments'
            )
        elif mode == self._modeclass.PAN_ZOOM:
            self.help = ''

        return mode

    @property
    def mode(self) -> str:
        """str: Interactive mode

        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        TRANSFORM allows for manipulation of the layer transform.
        """
        return str(self._mode)

    @mode.setter
    def mode(self, mode):
        mode = self._mode_setter_helper(mode)
        if mode == self._mode:
            return
        self._mode = mode

        self.events.mode(mode=str(mode))

    @classmethod
    def _basename(cls):
        return f'{cls.__name__}'

    @property
    def name(self):
        """str: Unique name of the layer."""
        return self._name

    @name.setter
    def name(self, name):
        if name == self.name:
            return
        if not name:
            name = self._basename()
        self._name = str(name)
        self.events.name()

    @property
    def metadata(self) -> dict:
        """Key/value map for user-stored data."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict) -> None:
        self._metadata.clear()
        self._metadata.update(value)

    @property
    def source(self):
        return self._source

    @property
    def loaded(self) -> bool:
        """Return True if this layer is fully loaded in memory.

        This base class says that layers are permanently in the loaded state.
        Derived classes that do asynchronous loading can override this.
        """
        return True

    @property
    def opacity(self):
        """float: Opacity value between 0.0 and 1.0."""
        return self._opacity

    @opacity.setter
    def opacity(self, opacity):
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(
                trans._(
                    'opacity must be between 0.0 and 1.0; got {opacity}',
                    deferred=True,
                    opacity=opacity,
                )
            )

        self._opacity = opacity
        self._update_thumbnail()
        self.events.opacity()

    @property
    def blending(self):
        """Blending mode: Determines how RGB and alpha values get mixed.

        Blending.OPAQUE
            Allows for only the top layer to be visible and corresponds to
            depth_test=True, cull_face=False, blend=False.
        Blending.TRANSLUCENT
            Allows for multiple layers to be blended with different opacity
            and corresponds to depth_test=True, cull_face=False,
            blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'),
            and blend_equation=('func_add').
        Blending.TRANSLUCENT_NO_DEPTH
          Allows for multiple layers to be blended with different opacity, but
          no depth testing is performed. Corresponds to ``depth_test=False``,
          cull_face=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'),
          and blend_equation=('func_add').
        Blending.ADDITIVE
            Allows for multiple layers to be blended together with
            different colors and opacity. Useful for creating overlays. It
            corresponds to depth_test=False, cull_face=False, blend=True,
            blend_func=('src_alpha', 'one'), and blend_equation=('func_add').
        Blending.MINIMUM
            Allows for multiple layers to be blended together such that
            the minimum of each RGB component and alpha are selected.
            Useful for creating overlays with inverted colormaps. It
            corresponds to depth_test=False, cull_face=False, blend=True,
            blend_equation=('min').
        """
        return str(self._blending)

    @blending.setter
    def blending(self, blending):
        self._blending = Blending(blending)
        self.events.blending()

    @property
    def visible(self) -> bool:
        """bool: Whether the visual is currently being displayed."""
        return self._visible

    @visible.setter
    def visible(self, visible: bool):
        self._visible = visible
        self.refresh()
        self.events.visible()

    @property
    def editable(self) -> bool:
        """bool: Whether the current layer data is editable from the viewer."""
        return self._editable

    @editable.setter
    def editable(self, editable: bool):
        if self._editable == editable:
            return
        self._editable = editable
        self._on_editable_changed()
        self.events.editable()

    def _reset_editable(self) -> None:
        """Reset this layer's editable state based on layer properties."""
        self.editable = True

    def _on_editable_changed(self) -> None:
        """Executes side-effects on this layer related to changes of the editable state."""

    @property
    def scale(self):
        """list: Anisotropy factors to scale data into world coordinates."""
        return self._transforms['data2physical'].scale

    @scale.setter
    def scale(self, scale):
        if scale is None:
            scale = [1] * self.ndim
        self._transforms['data2physical'].scale = np.array(scale)
        self._clear_extent()
        self.events.scale()

    @property
    def translate(self):
        """list: Factors to shift the layer by in units of world coordinates."""
        return self._transforms['data2physical'].translate

    @translate.setter
    def translate(self, translate):
        self._transforms['data2physical'].translate = np.array(translate)
        self._clear_extent()
        self.events.translate()

    @property
    def rotate(self):
        """array: Rotation matrix in world coordinates."""
        return self._transforms['data2physical'].rotate

    @rotate.setter
    def rotate(self, rotate):
        self._transforms['data2physical'].rotate = rotate
        self._clear_extent()
        self.events.rotate()

    @property
    def shear(self):
        """array: Shear matrix in world coordinates."""
        return self._transforms['data2physical'].shear

    @shear.setter
    def shear(self, shear):
        self._transforms['data2physical'].shear = shear
        self._clear_extent()
        self.events.shear()

    @property
    def affine(self):
        """napari.utils.transforms.Affine: Extra affine transform to go from physical to world coordinates."""
        return self._transforms['physical2world']

    @affine.setter
    def affine(self, affine):
        # Assignment by transform name is not supported by TransformChain and
        # EventedList, so use the integer index instead. For more details, see:
        # https://github.com/napari/napari/issues/3058
        self._transforms[2] = coerce_affine(
            affine, ndim=self.ndim, name='physical2world'
        )
        self._clear_extent()
        self.events.affine()

    @property
    def translate_grid(self):
        warnings.warn(
            trans._(
                "translate_grid will become private in v0.4.14. See Layer.translate or Layer.data_to_world() instead.",
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return self._translate_grid

    @translate_grid.setter
    def translate_grid(self, translate_grid):
        warnings.warn(
            trans._(
                "translate_grid will become private in v0.4.14. See Layer.translate or Layer.data_to_world() instead.",
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        self._translate_grid = translate_grid

    @property
    def _translate_grid(self):
        """list: Factors to shift the layer by."""
        return self._transforms['world2grid'].translate

    @_translate_grid.setter
    def _translate_grid(self, translate_grid):
        if np.all(self._translate_grid == translate_grid):
            return
        self._transforms['world2grid'].translate = np.array(translate_grid)
        self.events.translate()

    @property
    def _is_moving(self):
        return self._private_is_moving

    @_is_moving.setter
    def _is_moving(self, value):
        assert value in (True, False)
        if value:
            assert self._moving_coordinates is not None
        self._private_is_moving = value

    def _update_dims(self):
        """Update the dimensionality of transforms and slices when data changes."""
        ndim = self._get_ndim()

        old_ndim = self._ndim
        if old_ndim > ndim:
            keep_axes = range(old_ndim - ndim, old_ndim)
            self._transforms = self._transforms.set_slice(keep_axes)
        elif old_ndim < ndim:
            new_axes = range(ndim - old_ndim)
            self._transforms = self._transforms.expand_dims(new_axes)

        self._slice_input = self._slice_input.with_ndim(ndim)

        self._ndim = ndim

        self._clear_extent()

    @property
    @abstractmethod
    def data(self):
        # user writes own docstring
        raise NotImplementedError

    @data.setter
    @abstractmethod
    def data(self, data):
        raise NotImplementedError

    @property
    @abstractmethod
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        raise NotImplementedError

    @property
    def _extent_world(self) -> np.ndarray:
        """Range of layer in world coordinates.

        Returns
        -------
        extent_world : array, shape (2, D)
        """
        # Get full nD bounding box
        return get_extent_world(
            self._extent_data, self._data_to_world, self._array_like
        )

    @cached_property
    def extent(self) -> Extent:
        """Extent of layer in data and world coordinates."""
        extent_data = self._extent_data
        data_to_world = self._data_to_world
        extent_world = get_extent_world(
            extent_data, data_to_world, self._array_like
        )
        return Extent(
            data=extent_data,
            world=extent_world,
            step=abs(data_to_world.scale),
        )

    def _clear_extent(self):
        """Clears the cached extent.

        This should be called whenever this data or transform information
        changes, and should be called before any related events get emitted
        so that they use the updated extent values.
        """
        if 'extent' in self.__dict__:
            del self.extent
        self.events.extent()
        self.refresh()

    @property
    def _slice_indices(self):
        """(D, ) array: Slice indices in data coordinates."""
        if len(self._slice_input.not_displayed) == 0:
            # All dims are displayed dimensions
            return (slice(None),) * self.ndim
        return self._slice_input.data_indices(
            self._data_to_world.inverse,
            getattr(self, '_round_index', True),
        )

    @abstractmethod
    def _get_ndim(self):
        raise NotImplementedError

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
            'rotate': [list(r) for r in self.rotate],
            'shear': list(self.shear),
            'affine': self.affine.affine_matrix,
            'opacity': self.opacity,
            'blending': self.blending,
            'visible': self.visible,
            'experimental_clipping_planes': [
                plane.dict() for plane in self.experimental_clipping_planes
            ],
        }
        return base_dict

    @abstractmethod
    def _get_state(self):
        raise NotImplementedError

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
    def thumbnail(self, thumbnail):
        if 0 in thumbnail.shape:
            thumbnail = np.zeros(self._thumbnail_shape, dtype=np.uint8)
        if thumbnail.dtype != np.uint8:
            thumbnail = convert_to_uint8(thumbnail)

        padding_needed = np.subtract(self._thumbnail_shape, thumbnail.shape)
        pad_amounts = [(p // 2, (p + 1) // 2) for p in padding_needed]
        thumbnail = np.pad(thumbnail, pad_amounts, mode='constant')

        # blend thumbnail with opaque black background
        background = np.zeros(self._thumbnail_shape, dtype=np.uint8)
        background[..., 3] = 255

        f_dest = thumbnail[..., 3][..., None] / 255
        f_source = 1 - f_dest
        thumbnail = thumbnail * f_dest + background * f_source

        self._thumbnail = thumbnail.astype(np.uint8)
        self.events.thumbnail()

    @property
    def ndim(self):
        """int: Number of dimensions in the data."""
        return self._ndim

    @property
    def help(self):
        """str: displayed in status bar bottom right."""
        return self._help

    @help.setter
    def help(self, help_text):
        if help_text == self.help:
            return
        self._help = help_text
        self.events.help(help=help_text)

    @property
    def interactive(self) -> bool:
        warnings.warn(
            trans._(
                "Layer.interactive is deprecated since napari 0.4.18 and will be removed in 0.6.0. Please use Layer.mouse_pan and Layer.mouse_zoom instead"
            ),
            FutureWarning,
            stacklevel=2,
        )
        return self.mouse_pan or self.mouse_zoom

    @interactive.setter
    def interactive(self, interactive: bool):
        warnings.warn(
            trans._(
                "Layer.interactive is deprecated since napari 0.4.18 and will be removed in 0.6.0. Please use Layer.mouse_pan and Layer.mouse_zoom instead"
            ),
            FutureWarning,
            stacklevel=2,
        )
        with self.events.interactive.blocker():
            self.mouse_pan = interactive
        self.mouse_zoom = interactive

    @property
    def mouse_pan(self) -> bool:
        """bool: Determine if canvas interactive panning is enabled with the mouse."""
        return self._mouse_pan

    @mouse_pan.setter
    def mouse_pan(self, mouse_pan: bool):
        if mouse_pan == self._mouse_pan:
            return
        self._mouse_pan = mouse_pan
        self.events.mouse_pan(mouse_pan=mouse_pan)
        self.events.interactive(
            interactive=self.mouse_pan or self.mouse_zoom
        )  # Deprecated since 0.5.0

    @property
    def mouse_zoom(self) -> bool:
        """bool: Determine if canvas interactive zooming is enabled with the mouse."""
        return self._mouse_zoom

    @mouse_zoom.setter
    def mouse_zoom(self, mouse_zoom: bool):
        if mouse_zoom == self._mouse_zoom:
            return
        self._mouse_zoom = mouse_zoom
        self.events.mouse_zoom(mouse_zoom=mouse_zoom)
        self.events.interactive(
            interactive=self.mouse_pan or self.mouse_zoom
        )  # Deprecated since 0.5.0

    @property
    def cursor(self):
        """str: String identifying cursor displayed over canvas."""
        return self._cursor

    @cursor.setter
    def cursor(self, cursor):
        if cursor == self.cursor:
            return
        self._cursor = cursor
        self.events.cursor(cursor=cursor)

    @property
    def cursor_size(self):
        """int | None: Size of cursor if custom. None yields default size."""
        return self._cursor_size

    @cursor_size.setter
    def cursor_size(self, cursor_size):
        if cursor_size == self.cursor_size:
            return
        self._cursor_size = cursor_size
        self.events.cursor_size(cursor_size=cursor_size)

    @property
    def experimental_clipping_planes(self):
        return self._experimental_clipping_planes

    @experimental_clipping_planes.setter
    def experimental_clipping_planes(
        self,
        value: Union[
            dict,
            ClippingPlane,
            List[Union[ClippingPlane, dict]],
            ClippingPlaneList,
        ],
    ):
        self._experimental_clipping_planes.clear()
        if value is None:
            return

        if isinstance(value, (ClippingPlane, dict)):
            value = [value]
        for new_plane in value:
            plane = ClippingPlane()
            plane.update(new_plane)
            self._experimental_clipping_planes.append(plane)

    @property
    def bounding_box(self):
        return self._overlays['bounding_box']

    def set_view_slice(self):
        with self.dask_optimized_slicing():
            self._set_view_slice()

    @abstractmethod
    def _set_view_slice(self):
        raise NotImplementedError

    def _slice_dims(self, point=None, ndisplay=2, order=None):
        """Slice data with values from a global dims model.

        Note this will likely be moved off the base layer soon.

        Parameters
        ----------
        point : list
            Values of data to slice at in world coordinates.
        ndisplay : int
            Number of dimensions to be displayed.
        order : list of int
            Order of dimensions, where last `ndisplay` will be
            rendered in canvas.
        """
        point = (0,) * self.ndim if point is None else tuple(point)

        ndim = len(point)

        if order is None:
            order = tuple(range(ndim))

        # Correspondence between dimensions across all layers and
        # dimensions of this layer.
        point = point[-self.ndim :]
        order = tuple(
            self._world_to_layer_dims(world_dims=order, ndim_world=ndim)
        )

        slice_input = _SliceInput(
            ndisplay=ndisplay,
            point=point,
            order=order,
        )

        if self._slice_input == slice_input:
            return

        self._slice_input = slice_input

        # Update the point values
        self.refresh()
        self._reset_editable()

    @abstractmethod
    def _update_thumbnail(self):
        raise NotImplementedError

    @abstractmethod
    def _get_value(self, position):
        """Value of the data at a position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : tuple
            Value of the data.
        """
        raise NotImplementedError

    def get_value(
        self,
        position: Tuple[float],
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[List[int]] = None,
        world=False,
    ):
        """Value of the data at a position.

        If the layer is not visible, return None.

        Parameters
        ----------
        position : tuple of float
            Position in either data or world coordinates.
        view_direction : Optional[np.ndarray]
            A unit vector giving the direction of the ray in nD world coordinates.
            The default value is None.
        dims_displayed : Optional[List[int]]
            A list of the dimensions currently being displayed in the viewer.
            The default value is None.
        world : bool
            If True the position is taken to be in world coordinates
            and converted into data coordinates. False by default.

        Returns
        -------
        value : tuple, None
            Value of the data. If the layer is not visible return None.
        """
        if self.visible:
            if world:
                ndim_world = len(position)

                if dims_displayed is not None:
                    # convert the dims_displayed to the layer dims.This accounts
                    # for differences in the number of dimensions in the world
                    # dims versus the layer and for transpose and rolls.
                    dims_displayed = dims_displayed_world_to_layer(
                        dims_displayed,
                        ndim_world=ndim_world,
                        ndim_layer=self.ndim,
                    )
                position = self.world_to_data(position)

            if (dims_displayed is not None) and (view_direction is not None):
                if len(dims_displayed) == 2 or self.ndim == 2:
                    value = self._get_value(position=tuple(position))

                elif len(dims_displayed) == 3:
                    view_direction = self._world_to_data_ray(
                        list(view_direction)
                    )
                    start_point, end_point = self.get_ray_intersections(
                        position=position,
                        view_direction=view_direction,
                        dims_displayed=dims_displayed,
                        world=False,
                    )
                    value = self._get_value_3d(
                        start_point=start_point,
                        end_point=end_point,
                        dims_displayed=dims_displayed,
                    )
            else:
                value = self._get_value(position)

        else:
            value = None
        # This should be removed as soon as possible, it is still
        # used in Points and Shapes.
        self._value = value
        return value

    def _get_value_3d(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Union[
        float, int, None, Tuple[Union[float, int, None], Optional[int]]
    ]:
        """Get the layer data value along a ray

        Parameters
        ----------
        start_point : np.ndarray
            The start position of the ray used to interrogate the data.
        end_point : np.ndarray
            The end position of the ray used to interrogate the data.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the Viewer.

        Returns
        -------
        value
            The data value along the supplied ray.
        """

    def projected_distance_from_mouse_drag(
        self,
        start_position: np.ndarray,
        end_position: np.ndarray,
        view_direction: np.ndarray,
        vector: np.ndarray,
        dims_displayed: Union[List, np.ndarray],
    ):
        """Calculate the length of the projection of a line between two mouse
        clicks onto a vector (or array of vectors) in data coordinates.

        Parameters
        ----------
        start_position : np.ndarray
            Starting point of the drag vector in data coordinates
        end_position : np.ndarray
            End point of the drag vector in data coordinates
        view_direction : np.ndarray
            Vector defining the plane normal of the plane onto which the drag
            vector is projected.
        vector : np.ndarray
            (3,) unit vector or (n, 3) array thereof on which to project the drag
            vector from start_event to end_event. This argument is defined in data
            coordinates.
        dims_displayed : Union[List, np.ndarray]
            (3,) list of currently displayed dimensions

        Returns
        -------
        projected_distance : (1, ) or (n, ) np.ndarray of float
        """
        start_position = self._world_to_displayed_data(
            start_position, dims_displayed
        )
        end_position = self._world_to_displayed_data(
            end_position, dims_displayed
        )
        view_direction = self._world_to_displayed_data_ray(
            view_direction, dims_displayed
        )
        return drag_data_to_projected_distance(
            start_position, end_position, view_direction, vector
        )

    @contextmanager
    def block_update_properties(self):
        previous = self._update_properties
        self._update_properties = False
        try:
            yield
        finally:
            self._update_properties = previous

    def _set_highlight(self, force=False):
        """Render layer highlights when appropriate.

        Parameters
        ----------
        force : bool
            Bool that forces a redraw to occur when `True`.
        """

    def refresh(self, event=None):
        """Refresh all layer data based on current view slice."""
        if self.visible:
            self.set_view_slice()
            self.events.set_data()
            self._update_thumbnail()
            self._set_highlight(force=True)

    def world_to_data(self, position):
        """Convert from world coordinates to data coordinates.

        Parameters
        ----------
        position : tuple, list, 1D array
            Position in world coordinates. If longer then the
            number of dimensions of the layer, the later
            dimensions will be used.

        Returns
        -------
        tuple
            Position in data coordinates.
        """
        if len(position) >= self.ndim:
            coords = list(position[-self.ndim :])
        else:
            coords = [0] * (self.ndim - len(position)) + list(position)

        return tuple(self._transforms[1:].simplified.inverse(coords))

    def data_to_world(self, position):
        """Convert from data coordinates to world coordinates.

        Parameters
        ----------
        position : tuple, list, 1D array
            Position in data coordinates. If longer then the
            number of dimensions of the layer, the later
            dimensions will be used.

        Returns
        -------
        tuple
            Position in world coordinates.
        """
        if len(position) >= self.ndim:
            coords = list(position[-self.ndim :])
        else:
            coords = [0] * (self.ndim - len(position)) + list(position)

        return tuple(self._transforms[1:].simplified(coords))

    def _world_to_displayed_data(
        self, position: np.ndarray, dims_displayed: np.ndarray
    ) -> tuple:
        """Convert world to data coordinates for displayed dimensions only.

        Parameters
        ----------
        position : tuple, list, 1D array
            Position in world coordinates. If longer then the
            number of dimensions of the layer, the later
            dimensions will be used.
        dims_displayed : list, 1D array
            Indices of displayed dimensions of the data.

        Returns
        -------
        tuple
            Position in data coordinates for the displayed dimensions only
        """
        position_nd = self.world_to_data(position)
        position_ndisplay = np.asarray(position_nd)[dims_displayed]
        return tuple(position_ndisplay)

    @property
    def _data_to_world(self) -> Affine:
        """The transform from data to world coordinates.

        This affine transform is composed from the affine property and the
        other transform properties in the following order:

        affine * (rotate * shear * scale + translate)
        """
        return self._transforms[1:3].simplified

    def _world_to_data_ray(self, vector) -> tuple:
        """Convert a vector defining an orientation from world coordinates to data coordinates.
        For example, this would be used to convert the view ray.

        Parameters
        ----------
        vector : tuple, list, 1D array
            A vector in world coordinates.

        Returns
        -------
        tuple
            Vector in data coordinates.
        """
        p1 = np.asarray(self.world_to_data(vector))
        p0 = np.asarray(self.world_to_data(np.zeros_like(vector)))
        normalized_vector = (p1 - p0) / np.linalg.norm(p1 - p0)

        return tuple(normalized_vector)

    def _world_to_displayed_data_ray(
        self, vector_world: npt.ArrayLike, dims_displayed: List[int]
    ) -> np.ndarray:
        """Convert an orientation from world to displayed data coordinates.

        For example, this would be used to convert the view ray.

        Parameters
        ----------
        vector_world : tuple, list, 1D array
            A vector in world coordinates.

        Returns
        -------
        tuple
            Vector in data coordinates.
        """
        vector_data_nd = np.asarray(self._world_to_data_ray(vector_world))
        vector_data_ndisplay = vector_data_nd[dims_displayed]
        vector_data_ndisplay /= np.linalg.norm(vector_data_ndisplay)
        return vector_data_ndisplay

    def _world_to_layer_dims(
        self, *, world_dims: List[int], ndim_world: int
    ) -> List[int]:
        """Map world dimensions to layer dimensions while maintaining order.

        This is used to map dimensions from the full world space defined by ``Dims``
        to the subspace that a layer inhabits, so that those can be used to index the
        layer's data and associated coordinates.

        For example a world ``Dims.order`` of [2, 1, 0, 3] would map to [0, 1] for a
        layer with two dimensions and [1, 0, 2] for a layer with three dimensions
        as those correspond to the relative order of the last two and three world dimensions
        respectively.

        Parameters
        ----------
        world_dims : List[int]
            The world dimensions.
        ndim_world : int
            The number of dimensions in the world coordinate system.

        Returns
        -------
        List[int]
            The corresponding layer dimensions with the same ordering as the given world dimensions.
        """
        offset = ndim_world - self.ndim
        order = np.array(world_dims)
        if offset <= 0:
            return list(range(-offset)) + list(order - offset)

        return list(order[order >= offset] - offset)

    def _display_bounding_box(self, dims_displayed: np.ndarray):
        """An axis aligned (ndisplay, 2) bounding box around the data"""
        return self._extent_data[:, dims_displayed].T

    def click_plane_from_click_data(
        self,
        click_position: np.ndarray,
        view_direction: np.ndarray,
        dims_displayed: List,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate a (point, normal) plane parallel to the canvas in data
        coordinates, centered on the centre of rotation of the camera.

        Parameters
        ----------
        click_position : np.ndarray
            click position in world coordinates from mouse event.
        view_direction : np.ndarray
            view direction in world coordinates from mouse event.
        dims_displayed : List
            dimensions of the data array currently in view.

        Returns
        -------
        click_plane : Tuple[np.ndarray, np.ndarray]
            tuple of (plane_position, plane_normal) in data coordinates.
        """
        click_position = np.asarray(click_position)
        view_direction = np.asarray(view_direction)
        plane_position = self.world_to_data(click_position)[dims_displayed]
        plane_normal = self._world_to_data_ray(view_direction)[dims_displayed]
        return plane_position, plane_normal

    def get_ray_intersections(
        self,
        position: List[float],
        view_direction: np.ndarray,
        dims_displayed: List[int],
        world: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
        """Get the start and end point for the ray extending
        from a point through the data bounding box.

        Parameters
        ----------
        position
            the position of the point in nD coordinates. World vs. data
            is set by the world keyword argument.
        view_direction : np.ndarray
            a unit vector giving the direction of the ray in nD coordinates.
            World vs. data is set by the world keyword argument.
        dims_displayed
            a list of the dimensions currently being displayed in the viewer.
        world : bool
            True if the provided coordinates are in world coordinates.
            Default value is True.

        Returns
        -------
        start_point : np.ndarray
            The point on the axis-aligned data bounding box that the cursor click
            intersects with. This is the point closest to the camera.
            The point is the full nD coordinates of the layer data.
            If the click does not intersect the axis-aligned data bounding box,
            None is returned.
        end_point : np.ndarray
            The point on the axis-aligned data bounding box that the cursor click
            intersects with. This is the point farthest from the camera.
            The point is the full nD coordinates of the layer data.
            If the click does not intersect the axis-aligned data bounding box,
            None is returned.
        """
        if len(dims_displayed) != 3:
            return None, None

        # create the bounding box in data coordinates
        bounding_box = self._display_bounding_box(dims_displayed)

        start_point, end_point = self._get_ray_intersections(
            position=position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
            bounding_box=bounding_box,
        )
        return start_point, end_point

    def _get_offset_data_position(self, position: List[float]) -> List[float]:
        """Adjust position for offset between viewer and data coordinates."""
        return position

    def _get_ray_intersections(
        self,
        position: List[float],
        view_direction: np.ndarray,
        dims_displayed: List[int],
        world: bool = True,
        bounding_box: Optional[np.ndarray] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
        """Get the start and end point for the ray extending
        from a point through the data bounding box.

        Parameters
        ----------
        position
            the position of the point in nD coordinates. World vs. data
            is set by the world keyword argument.
        view_direction : np.ndarray
            a unit vector giving the direction of the ray in nD coordinates.
            World vs. data is set by the world keyword argument.
        dims_displayed
            a list of the dimensions currently being displayed in the viewer.
        world : bool
            True if the provided coordinates are in world coordinates.
            Default value is True.
        bounding_box : np.ndarray
            A (2, 3) bounding box around the data currently in view

        Returns
        -------
        start_point : np.ndarray
            The point on the axis-aligned data bounding box that the cursor click
            intersects with. This is the point closest to the camera.
            The point is the full nD coordinates of the layer data.
            If the click does not intersect the axis-aligned data bounding box,
            None is returned.
        end_point : np.ndarray
            The point on the axis-aligned data bounding box that the cursor click
            intersects with. This is the point farthest from the camera.
            The point is the full nD coordinates of the layer data.
            If the click does not intersect the axis-aligned data bounding box,
            None is returned."""
        # get the view direction and click position in data coords
        # for the displayed dimensions only
        if world is True:
            view_dir = self._world_to_displayed_data_ray(
                view_direction, dims_displayed
            )
            click_pos_data = self._world_to_displayed_data(
                position, dims_displayed
            )
        else:
            # adjust for any offset between viewer and data coordinates
            position = self._get_offset_data_position(position)

            view_dir = np.asarray(view_direction)[dims_displayed]
            click_pos_data = np.asarray(position)[dims_displayed]

        # Determine the front and back faces
        front_face_normal, back_face_normal = find_front_back_face(
            click_pos_data, bounding_box, view_dir
        )
        if front_face_normal is None and back_face_normal is None:
            # click does not intersect the data bounding box
            return None, None

        # Calculate ray-bounding box face intersections
        start_point_displayed_dimensions = (
            intersect_line_with_axis_aligned_bounding_box_3d(
                click_pos_data, view_dir, bounding_box, front_face_normal
            )
        )
        end_point_displayed_dimensions = (
            intersect_line_with_axis_aligned_bounding_box_3d(
                click_pos_data, view_dir, bounding_box, back_face_normal
            )
        )

        # add the coordinates for the axes not displayed
        start_point = np.asarray(position)
        start_point[dims_displayed] = start_point_displayed_dimensions
        end_point = np.asarray(position)
        end_point[dims_displayed] = end_point_displayed_dimensions

        return start_point, end_point

    def _update_draw(
        self, scale_factor, corner_pixels_displayed, shape_threshold
    ):
        """Update canvas scale and corner values on draw.

        For layer multiscale determining if a new resolution level or tile is
        required.

        Parameters
        ----------
        scale_factor : float
            Scale factor going from canvas to world coordinates.
        corner_pixels_displayed : array, shape (2, 2)
            Coordinates of the top-left and bottom-right canvas pixels in
            world coordinates.
        shape_threshold : tuple
            Requested shape of field of view in data coordinates.
        """
        self.scale_factor = scale_factor

        displayed_axes = self._slice_input.displayed

        # we need to compute all four corners to compute a complete,
        # data-aligned bounding box, because top-left/bottom-right may not
        # remain top-left and bottom-right after transformations.
        all_corners = list(itertools.product(*corner_pixels_displayed.T))
        # Note that we ignore the first transform which is tile2data
        data_corners = (
            self._transforms[1:]
            .simplified.set_slice(displayed_axes)
            .inverse(all_corners)
        )

        # find the maximal data-axis-aligned bounding box containing all four
        # canvas corners and round them to ints
        data_bbox = np.stack(
            [np.min(data_corners, axis=0), np.max(data_corners, axis=0)]
        )
        data_bbox_int = np.stack(
            [np.floor(data_bbox[0]), np.ceil(data_bbox[1])]
        ).astype(int)

        if self._slice_input.ndisplay == 2 and self.multiscale:
            level, scaled_corners = compute_multiscale_level_and_corners(
                data_bbox_int,
                shape_threshold,
                self.downsample_factors[:, displayed_axes],
            )
            corners = np.zeros((2, self.ndim), dtype=int)
            # The corner_pixels attribute stores corners in the data
            # space of the selected level. Using the level's data
            # shape only works for images, but that's the only case we
            # handle now and downsample_factors is also only on image layers.
            max_coords = np.take(self.data[level].shape, displayed_axes)
            corners[:, displayed_axes] = np.clip(scaled_corners, 0, max_coords)
            display_shape = tuple(
                corners[1, displayed_axes] - corners[0, displayed_axes]
            )
            if any(s == 0 for s in display_shape):
                return
            if self.data_level != level or not np.all(
                self.corner_pixels == corners
            ):
                self._data_level = level
                self.corner_pixels = corners
                self.refresh()

        else:
            # The stored corner_pixels attribute must contain valid indices.
            corners = np.zeros((2, self.ndim), dtype=int)
            # Some empty layers (e.g. Points) may have a data extent that only
            # contains nans, in which case the integer valued corner pixels
            # cannot be meaningfully set.
            displayed_extent = self.extent.data[:, displayed_axes]
            if not np.all(np.isnan(displayed_extent)):
                data_bbox_clipped = np.clip(
                    data_bbox_int, displayed_extent[0], displayed_extent[1]
                )
                corners[:, displayed_axes] = data_bbox_clipped
            self.corner_pixels = corners

    def _get_source_info(self):
        components = {}
        if self.source.reader_plugin:
            components['layer_base'] = os.path.basename(self.source.path or '')
            components['source_type'] = 'plugin'
            try:
                components['plugin'] = pm.get_manifest(
                    self.source.reader_plugin
                ).display_name
            except KeyError:
                components['plugin'] = self.source.reader_plugin
            return components

        if self.source.sample:
            components['layer_base'] = self.name
            components['source_type'] = 'sample'
            try:
                components['plugin'] = pm.get_manifest(
                    self.source.sample[0]
                ).display_name
            except KeyError:
                components['plugin'] = self.source.sample[0]
            return components

        if self.source.widget:
            components['layer_base'] = self.name
            components['source_type'] = 'widget'
            components['plugin'] = self.source.widget._function.__name__
            return components

        components['layer_base'] = self.name
        components['source_type'] = ''
        components['plugin'] = ''
        return components

    def get_source_str(self):
        source_info = self._get_source_info()

        return (
            source_info['layer_base']
            + ', '
            + source_info['source_type']
            + ' : '
            + source_info['plugin']
        )

    def get_status(
        self,
        position: Optional[Tuple[float, ...]] = None,
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[List[int]] = None,
        world=False,
    ):
        """
        Status message information of the data at a coordinate position.

        Parameters
        ----------
        position : tuple of float
            Position in either data or world coordinates.
        view_direction : Optional[np.ndarray]
            A unit vector giving the direction of the ray in nD world coordinates.
            The default value is None.
        dims_displayed : Optional[List[int]]
            A list of the dimensions currently being displayed in the viewer.
            The default value is None.
        world : bool
            If True the position is taken to be in world coordinates
            and converted into data coordinates. False by default.

        Returns
        -------
        source_info : dict
            Dictionary containing a information that can be used as a status update.
        """
        if position is not None:
            value = self.get_value(
                position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=world,
            )
        else:
            value = None

        source_info = self._get_source_info()
        source_info['coordinates'] = generate_layer_coords_status(
            position[-self.ndim :], value
        )
        return source_info

    def _get_tooltip_text(
        self,
        position,
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[List[int]] = None,
        world: bool = False,
    ):
        """
        tooltip message of the data at a coordinate position.

        Parameters
        ----------
        position : tuple
            Position in either data or world coordinates.
        view_direction : Optional[np.ndarray]
            A unit vector giving the direction of the ray in nD world coordinates.
            The default value is None.
        dims_displayed : Optional[List[int]]
            A list of the dimensions currently being displayed in the viewer.
            The default value is None.
        world : bool
            If True the position is taken to be in world coordinates
            and converted into data coordinates. False by default.

        Returns
        -------
        msg : string
            String containing a message that can be used as a tooltip.
        """
        return ""

    def save(self, path: str, plugin: Optional[str] = None) -> List[str]:
        """Save this layer to ``path`` with default (or specified) plugin.

        Parameters
        ----------
        path : str
            A filepath, directory, or URL to open.  Extensions may be used to
            specify output format (provided a plugin is available for the
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
        from napari.plugins.io import save_layers

        return save_layers(path, [self], plugin=plugin)

    def _on_selection(self, selected: bool):
        # This method is a temporary workaround to the fact that the Points
        # layer needs to know when its selection state changes so that it can
        # update the highlight state.  This, along with the events.select and
        # events.deselect emitters, (and the LayerList._on_selection_event
        # method) can be removed once highlighting logic has been removed from
        # the layer model.
        if selected:
            self.events.select()
        else:
            self.events.deselect()

    @classmethod
    def create(
        cls, data, meta: dict = None, layer_type: Optional[str] = None
    ) -> Layer:
        """Create layer from `data` of type `layer_type`.

        Primarily intended for usage by reader plugin hooks and creating a
        layer from an unwrapped layer data tuple.

        Parameters
        ----------
        data : Any
            Data in a format that is valid for the corresponding `layer_type`.
        meta : dict, optional
            Dict of keyword arguments that will be passed to the corresponding
            layer constructor.  If any keys in `meta` are not valid for the
            corresponding layer type, an exception will be raised.
        layer_type : str
            Type of layer to add. Must be the (case insensitive) name of a
            Layer subclass.  If not provided, the layer is assumed to
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

        >>> data = (
        ...     np.random.random((10, 2)) * 20,
        ...     {'face_color': 'blue'},
        ...     'points',
        ... )
        >>> Layer.create(*data)

        """
        from napari import layers
        from napari.layers.image._image_utils import guess_labels

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

        Cls = getattr(layers, layer_type.title())

        try:
            return Cls(data, **(meta or {}))
        except Exception as exc:  # noqa: BLE001
            if 'unexpected keyword argument' not in str(exc):
                raise

            bad_key = str(exc).split('keyword argument ')[-1]
            raise TypeError(
                trans._(
                    "_add_layer_from_data received an unexpected keyword argument ({bad_key}) for layer type {layer_type}",
                    deferred=True,
                    bad_key=bad_key,
                    layer_type=layer_type,
                )
            ) from exc


mgui.register_type(type_=List[Layer], return_callback=add_layers_to_viewer)
