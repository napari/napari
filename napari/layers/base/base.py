import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from contextlib import contextmanager
from typing import List, Optional

import numpy as np

from ...utils.dask_utils import configure_dask
from ...utils.events import EmitterGroup, Event
from ...utils.key_bindings import KeymapProvider
from ...utils.misc import ROOT_DIR
from ...utils.naming import magic_name
from ...utils.status_messages import format_float, status_format
from ...utils.transforms import Affine, TransformChain
from ..utils.layer_utils import (
    compute_multiscale_level_and_corners,
    convert_to_uint8,
)
from ._base_constants import Blending

Extent = namedtuple('Extent', 'data world step')


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
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine: n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a lenght N translation vector and a 1 or a napari
        AffineTransform object. If provided then translate, scale, rotate, and
        shear values are ignored.
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
    opacity : float
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
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine: n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a lenght N translation vector and a 1 or a napari
        AffineTransform object. If provided then translate, scale, rotate, and
        shear values are ignored.
    multiscale : bool
        Whether the data is multiscale or not. Multiscale data is
        represented by a list of data objects and should go from largest to
        smallest.
    z_index : int
        Depth of the layer visual relative to other visuals in the scenecanvas.
    coordinates : tuple of float
        Cursor position in data coordinates.
    corner_pixels : array
        Coordinates of the top-left and bottom-right canvas pixels in the data
        coordinates of each layer. For multiscale data the coordinates are in
        the space of the currently viewed data level, not the highest resolution
        level.
    position : tuple
        Cursor position in world coordinates.
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
        * `_extent_data`: property
        * `data` property (setter & getter)

    May define the following:
        * `_set_view_slice()`: called to set currently viewed slice
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
        rotate=None,
        shear=None,
        affine=None,
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
        self._cursor_size = 1
        self._interactive = True
        self._value = None
        self.scale_factor = 1
        self.multiscale = multiscale

        self._ndim = ndim
        self._ndisplay = 2
        self._dims_order = list(range(ndim))

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

        # First create the `data2world` transform from the input parameters
        if affine is None:
            if scale is None:
                scale = [1] * ndim
            if translate is None:
                translate = [0] * ndim
            data2world_transform = Affine(
                scale,
                translate,
                rotate=rotate,
                shear=shear,
                name='data2world',
            )
        elif isinstance(affine, np.ndarray) or isinstance(affine, list):
            data2world_transform = Affine(
                affine_matrix=np.array(affine), name='data2world',
            )
        elif isinstance(affine, Affine):
            affine.name = 'data2world'
            data2world_transform = affine
        else:
            raise TypeError(
                (
                    'affine input not recognized. '
                    'must be either napari.utils.transforms.Affine, '
                    f'ndarray, or None. Got {type(affine)}'
                )
            )

        self._transforms = TransformChain(
            [
                Affine(np.ones(ndim), np.zeros(ndim), name='tile2data'),
                data2world_transform,
                Affine(np.ones(ndim), np.zeros(ndim), name='world2grid'),
            ]
        )

        self._position = (0,) * ndim
        self._dims_point = [0] * ndim
        self.corner_pixels = np.zeros((2, ndim), dtype=int)
        self._editable = True

        self._thumbnail_shape = (32, 32, 4)
        self._thumbnail = np.zeros(self._thumbnail_shape, dtype=np.uint8)
        self._update_properties = True
        self._name = ''
        self.events = EmitterGroup(
            source=self,
            auto_connect=False,
            refresh=Event,
            set_data=Event,
            blending=Event,
            opacity=Event,
            visible=Event,
            select=Event,
            deselect=Event,
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
            interactive=Event,
            cursor=Event,
            cursor_size=Event,
            editable=Event,
            loaded=Event,
            _ndisplay=Event,
        )
        self.name = name

        self.mouse_move_callbacks = []
        self.mouse_drag_callbacks = []
        self.mouse_wheel_callbacks = []
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

    @property
    def loaded(self) -> bool:
        """Return True if this layer is fully loaded in memory.

        This base class says that layers are permanently in the loaded state.
        Derived classes that do asynchronous loading can override this.
        """
        return True

    @name.setter
    def name(self, name):
        if name == self.name:
            return
        if not name:
            name = self._basename()
        self._name = name
        self.events.name()

    @property
    def opacity(self):
        """float: Opacity value between 0.0 and 1.0.
        """
        return self._opacity

    @opacity.setter
    def opacity(self, opacity):
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(
                'opacity must be between 0.0 and 1.0; ' f'got {opacity}'
            )

        self._opacity = opacity
        self._update_thumbnail()
        self.status = format_float(self.opacity)
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
            blend=True, blend_func=('src_alpha', 'one_minus_src_alpha').
        Blending.ADDITIVE
            Allows for multiple layers to be blended together with
            different colors and opacity. Useful for creating overlays. It
            corresponds to depth_test=False, cull_face=False, blend=True,
            blend_func=('src_alpha', 'one').
        """
        return str(self._blending)

    @blending.setter
    def blending(self, blending):
        self._blending = Blending(blending)
        self.events.blending()

    @property
    def visible(self):
        """bool: Whether the visual is currently being displayed."""
        return self._visible

    @visible.setter
    def visible(self, visibility):
        self._visible = visibility
        self.refresh()
        self.events.visible()
        if self.visible:
            self.editable = self._set_editable()
        else:
            self.editable = False

    @property
    def editable(self):
        """bool: Whether the current layer data is editable from the viewer."""
        return self._editable

    @editable.setter
    def editable(self, editable):
        if self._editable == editable:
            return
        self._editable = editable
        self._set_editable(editable=editable)
        self.events.editable()

    @property
    def scale(self):
        """list: Anisotropy factors to scale data into world coordinates."""
        return self._transforms['data2world'].scale

    @scale.setter
    def scale(self, scale):
        self._transforms['data2world'].scale = np.array(scale)
        self._update_dims()
        self.events.scale()

    @property
    def translate(self):
        """list: Factors to shift the layer by in units of world coordinates."""
        return self._transforms['data2world'].translate

    @translate.setter
    def translate(self, translate):
        self._transforms['data2world'].translate = np.array(translate)
        self._update_dims()
        self.events.translate()

    @property
    def rotate(self):
        """array: Rotation matrix in world coordinates."""
        return self._transforms['data2world'].rotate

    @rotate.setter
    def rotate(self, rotate):
        self._transforms['data2world'].rotate = rotate
        self._update_dims()
        self.events.rotate()

    @property
    def shear(self):
        """array: Sheer matrix in world coordinates."""
        return self._transforms['data2world'].shear

    @shear.setter
    def shear(self, shear):
        self._transforms['data2world'].shear = shear
        self._update_dims()
        self.events.shear()

    @property
    def affine(self):
        """napari.utils.transforms.Affine: Affine transform."""
        return self._transforms['data2world']

    @affine.setter
    def affine(self, affine):
        if isinstance(affine, np.ndarray) or isinstance(affine, list):
            self._transforms['data2world'].affine_matrix = np.array(affine)
        elif isinstance(affine, Affine):
            affine.name = 'data2world'
            self._transforms['data2world'] = affine
        else:
            raise TypeError(
                (
                    'affine input not recognized. '
                    'must be either napari.utils.transforms.Affine '
                    f'or ndarray. Got {type(affine)}'
                )
            )
        self._update_dims()
        self.events.affine()

    @property
    def translate_grid(self):
        """list: Factors to shift the layer by."""
        return self._transforms['world2grid'].translate

    @translate_grid.setter
    def translate_grid(self, translate_grid):
        if np.all(self.translate_grid == translate_grid):
            return
        self._transforms['world2grid'].translate = np.array(translate_grid)
        self.events.translate()

    @property
    def position(self):
        """tuple: Cursor position in world slice coordinates."""
        return self._position

    @position.setter
    def position(self, position):
        _position = position[-self.ndim :]
        if self._position == _position:
            return
        self._position = _position
        self._update_value_and_status()

    @property
    def _dims_displayed(self):
        """To be removed displayed dimensions."""
        # Ultimately we aim to remove all slicing information from the layer
        # itself so that layers can be sliced in different ways for multiple
        # canvas. See https://github.com/napari/napari/pull/1919#issuecomment-738585093
        # for additional discussion.
        return self._dims_order[-self._ndisplay :]

    @property
    def _dims_not_displayed(self):
        """To be removed not displayed dimensions."""
        # Ultimately we aim to remove all slicing information from the layer
        # itself so that layers can be sliced in different ways for multiple
        # canvas. See https://github.com/napari/napari/pull/1919#issuecomment-738585093
        # for additional discussion.
        return self._dims_order[: -self._ndisplay]

    @property
    def _dims_displayed_order(self):
        """To be removed order of displayed dimensions."""
        # Ultimately we aim to remove all slicing information from the layer
        # itself so that layers can be sliced in different ways for multiple
        # canvas. See https://github.com/napari/napari/pull/1919#issuecomment-738585093
        # for additional discussion.
        order = np.array(self._dims_displayed)
        order[np.argsort(order)] = list(range(len(order)))
        return tuple(order)

    def _update_dims(self, event=None):
        """Updates dims model, which is useful after data has been changed."""
        ndim = self._get_ndim()

        old_ndim = self._ndim
        if old_ndim > ndim:
            keep_axes = range(old_ndim - ndim, old_ndim)
            self._transforms = self._transforms.set_slice(keep_axes)
            self._dims_point = self._dims_point[-ndim:]
            arr = np.array(self._dims_order[-ndim:])
            arr[np.argsort(arr)] = range(len(arr))
            self._dims_order = arr.tolist()
            self._position = self._position[-ndim:]
        elif old_ndim < ndim:
            new_axes = range(ndim - old_ndim)
            self._transforms = self._transforms.expand_dims(new_axes)
            self._dims_point = [0] * (ndim - old_ndim) + self._dims_point
            self._dims_order = list(range(ndim - old_ndim)) + [
                o + ndim - old_ndim for o in self._dims_order
            ]
            self._position = (0,) * (ndim - old_ndim) + self._position

        self._ndim = ndim

        self.refresh()
        self._update_value_and_status()

    @property
    @abstractmethod
    def data(self):
        # user writes own docstring
        raise NotImplementedError()

    @data.setter
    @abstractmethod
    def data(self, data):
        raise NotImplementedError()

    @property
    @abstractmethod
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        raise NotImplementedError()

    @property
    def _extent_world(self) -> np.ndarray:
        """Range of layer in world coordinates.

        Returns
        -------
        extent_world : array, shape (2, D)
        """
        # Get full nD bounding box
        data_extent = self._extent_data
        D = data_extent.shape[1]
        full_data_extent = np.array(np.meshgrid(*data_extent.T)).T.reshape(
            -1, D
        )
        full_world_extent = self._transforms['data2world'](full_data_extent)
        world_extent = np.array(
            [
                np.min(full_world_extent, axis=0),
                np.max(full_world_extent, axis=0),
            ]
        )
        return world_extent

    @property
    def extent(self) -> Extent:
        """Extent of layer in data and world coordinates."""
        return Extent(
            data=self._extent_data,
            world=self._extent_world,
            step=abs(self.scale),
        )

    @property
    def _slice_indices(self):
        """(D, ) array: Slice indices in data coordinates."""
        inv_transform = self._transforms['data2world'].inverse

        if self.ndim > self._ndisplay:
            # Subspace spanned by non displayed dimensions
            non_displayed_subspace = np.zeros(self.ndim)
            for d in self._dims_not_displayed:
                non_displayed_subspace[d] = 1
            # Map subspace through inverse transform, ignoring translation
            mapped_nd_subspace = inv_transform(
                non_displayed_subspace
            ) - inv_transform(np.zeros(self.ndim))
            # Look at displayed subspace
            displayed_mapped_subspace = [
                mapped_nd_subspace[d] for d in self._dims_displayed
            ]
            # Check that displayed subspace is null
            if not np.allclose(displayed_mapped_subspace, 0):
                warnings.warn(
                    'Non-orthogonal slicing is being requested, but'
                    ' is not fully supported. Data is displayed without'
                    ' applying an out-of-slice rotation or shear component.',
                    category=UserWarning,
                )

        slice_inv_transform = inv_transform.set_slice(self._dims_not_displayed)

        world_pts = [self._dims_point[ax] for ax in self._dims_not_displayed]
        data_pts = slice_inv_transform(world_pts)
        # A round is taken to convert these values to slicing integers
        data_pts = np.round(data_pts).astype(int)

        indices = [slice(None)] * self.ndim
        for i, ax in enumerate(self._dims_not_displayed):
            indices[ax] = data_pts[i]

        return tuple(indices)

    @abstractmethod
    def _get_ndim(self):
        raise NotImplementedError()

    def _set_editable(self, editable=None):
        if editable is None:
            self.editable = True

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
    def thumbnail(self, thumbnail):
        if 0 in thumbnail.shape:
            thumbnail = np.zeros(self._thumbnail_shape, dtype=np.uint8)
        if thumbnail.dtype != np.uint8:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
    def selected(self):
        """bool: Whether this layer is selected or not."""
        return self._selected

    @selected.setter
    def selected(self, selected):
        if selected == self.selected:
            return
        self._selected = selected

        if selected:
            self.events.select()
        else:
            self.events.deselect()

    @property
    def status(self):
        """str: displayed in status bar bottom left."""
        return self._status

    @status.setter
    def status(self, status):
        if status == self.status:
            return
        self.events.status(status=status)
        self._status = status

    @property
    def help(self):
        """str: displayed in status bar bottom right."""
        return self._help

    @help.setter
    def help(self, help):
        if help == self.help:
            return
        self.events.help(help=help)
        self._help = help

    @property
    def interactive(self):
        """bool: Determine if canvas pan/zoom interactivity is enabled."""
        return self._interactive

    @interactive.setter
    def interactive(self, interactive):
        if interactive == self.interactive:
            return
        self.events.interactive(interactive=interactive)
        self._interactive = interactive

    @property
    def cursor(self):
        """str: String identifying cursor displayed over canvas."""
        return self._cursor

    @cursor.setter
    def cursor(self, cursor):
        if cursor == self.cursor:
            return
        self.events.cursor(cursor=cursor)
        self._cursor = cursor

    @property
    def cursor_size(self):
        """int | None: Size of cursor if custom. None yields default size."""
        return self._cursor_size

    @cursor_size.setter
    def cursor_size(self, cursor_size):
        if cursor_size == self.cursor_size:
            return
        self.events.cursor_size(cursor_size=cursor_size)
        self._cursor_size = cursor_size

    def set_view_slice(self):
        with self.dask_optimized_slicing():
            self._set_view_slice()

    @abstractmethod
    def _set_view_slice(self):
        raise NotImplementedError()

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
        if point is None:
            ndim = self.ndim
        else:
            ndim = len(point)

        if order is None:
            order = list(range(ndim))

        # adjust the order of the global dims based on the number of
        # dimensions that a layer has - for example a global order of
        # [2, 1, 0, 3] -> [0, 1] for a layer that only has two dimensions
        # or -> [1, 0, 2] for a layer with three as that corresponds to
        # the relative order of the last two and three dimensions
        # respectively
        offset = ndim - self.ndim
        order = np.array(order)
        if offset <= 0:
            order = list(range(-offset)) + list(order - offset)
        else:
            order = list(order[order >= offset] - offset)

        if point is None:
            point = [0] * ndim
            nd = min(self.ndim, ndisplay)
            for i in order[-nd:]:
                point[i] = slice(None)
        else:
            point = list(point)

        # If no slide data has changed, then do nothing
        if (
            np.all(order == self._dims_order)
            and ndisplay == self._ndisplay
            and np.all(point[offset:] == self._dims_point)
        ):
            return

        self._dims_order = order
        if self._ndisplay != ndisplay:
            self._ndisplay = ndisplay
            self.events._ndisplay()

        # Update the point values
        self._dims_point = point[offset:]
        self._update_dims()
        self._set_editable()

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
            self.events.set_data()
            self._update_thumbnail()
            self._update_value_and_status()
            self._set_highlight(force=True)

    @property
    def coordinates(self):
        """Cursor position in data coordinates."""
        # Note we ignore the first transform which is tile2data
        return tuple(self._transforms[1:].simplified.inverse(self.position))

    def _update_value_and_status(self):
        """Update value and status message."""
        self._value = self.get_value()
        self.status = self.get_message()

    def _update_draw(self, scale_factor, corner_pixels, shape_threshold):
        """Update canvas scale and corner values on draw.
        For layer multiscale determing if a new resolution level or tile is
        required.
        Parameters
        ----------
        scale_factor : float
            Scale factor going from canvas to world coordinates.
        corner_pixels : array
            Coordinates of the top-left and bottom-right canvas pixels in the
            world coordinates.
        shape_threshold : tuple
            Requested shape of field of view in data coordinates.
        """
        # Note we ignore the first transform which is tile2data
        data_corners = self._transforms[1:].simplified.inverse(corner_pixels)

        self.scale_factor = scale_factor

        # Round and clip data corners
        data_corners = np.array(
            [np.floor(data_corners[0]), np.ceil(data_corners[1])]
        ).astype(int)
        data_corners = np.clip(
            data_corners, self.extent.data[0], self.extent.data[1]
        )

        if self._ndisplay == 2 and self.multiscale:
            level, displayed_corners = compute_multiscale_level_and_corners(
                data_corners[:, self._dims_displayed],
                shape_threshold,
                self.downsample_factors[:, self._dims_displayed],
            )
            corners = np.zeros((2, self.ndim))
            corners[:, self._dims_displayed] = displayed_corners
            corners = corners.astype(int)
            if self.data_level != level or not np.all(
                self.corner_pixels == corners
            ):
                self._data_level = level
                self.corner_pixels = corners
                self.refresh()

        else:
            self.corner_pixels = data_corners

    @property
    def displayed_coordinates(self):
        """list: List of currently displayed coordinates."""
        coordinates = self.coordinates
        return [coordinates[i] for i in self._dims_displayed]

    def get_message(self):
        """Generate a status message based on the coordinates and value

        Returns
        -------
        msg : string
            String containing a message that can be used as a status update.
        """
        full_coord = np.round(self.coordinates).astype(int)

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
        from ...plugins.io import save_layers

        return save_layers(path, [self], plugin=plugin)
