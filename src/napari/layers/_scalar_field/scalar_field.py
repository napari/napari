from __future__ import annotations

import types
from abc import ABC, abstractmethod
from contextlib import nullcontext
from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy import typing as npt

from napari.layers._data_protocols import LayerDataProtocol
from napari.layers._multiscale_data import MultiScaleData
from napari.layers._scalar_field._slice import (
    _ScalarFieldSliceRequest,
    _ScalarFieldSliceResponse,
)
from napari.layers.base import Layer, _LayerSlicingState
from napari.layers.image._image_constants import Interpolation, VolumeDepiction
from napari.layers.image._image_mouse_bindings import (
    move_plane_along_normal as plane_drag_callback,
    set_plane_position as plane_double_click_callback,
)
from napari.layers.image._image_utils import guess_multiscale
from napari.layers.utils._slice_input import (
    _SliceInput,
    _ThickNDSlice,
)
from napari.layers.utils.layer_utils import (
    compute_multiscale_level_and_corners,
)
from napari.layers.utils.plane import SlicingPlane
from napari.types import LayerDataType
from napari.utils._dask_utils import DaskIndexer
from napari.utils._dtype import normalize_dtype
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.utils.events import Event
from napari.utils.events.event import WarningEmitter
from napari.utils.events.event_utils import connect_no_arg
from napari.utils.geometry import clamp_point_to_bounding_box
from napari.utils.naming import magic_name
from napari.utils.transforms import Affine
from napari.utils.translations import trans

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from napari.components import Dims


__all__ = ('ScalarFieldBase',)

#: Lower bound for the per-axis extent of 3D sub-volume tiles, so extreme
#: zooms do not thrash on tiny slices.
_MIN_TILE_EXTENT_3D = 32

#: 3D sub-volume tile extents are rounded up to a multiple of this, so
#: consecutive camera poses produce tiles with repeating shapes. The
#: viewport-derived extent otherwise jitters by a few voxels per pose,
#: and every new texture shape costs a GPU (re)allocation — a pipeline
#: synchronization on slow drivers — and defeats texture reuse/pooling.
_TILE_EXTENT_QUANTUM_3D = 32


def _make_level_materializer(
    data: MultiScaleData,
) -> Callable[[int], np.ndarray]:
    """Return a cached function that materializes one level of *data* at a time.

    The returned callable accepts a ``level`` integer and returns
    ``np.asarray(data[level])``, caching the last result so repeated slice
    requests at the same level avoid redundant data fetches.  Replacing
    ``data`` by constructing a new materializer automatically abandons the old
    cache without an explicit ``cache_clear()``.
    """

    @lru_cache(maxsize=1)
    def _materializer(level: int) -> np.ndarray:
        return np.asarray(data[level])

    return _materializer


# It is important to contain at least one abstractmethod to properly exclude this class
# in creating NAMES set inside of napari.layers.__init__
# Mixin must come before Layer
class ScalarFieldBase(Layer, ABC):
    """Base class for volumetric layers.

    Parameters
    ----------
    data : array or list of array
        Image data. Can be N >= 2 dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a
        list and arrays are decreasing in shape then the data is treated as
        a multiscale image. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    axis_labels : tuple of str, optional
        Dimension names of the layer data.
        If not provided, axis_labels will be set to (..., '-2', '-1').
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', 'translucent_no_depth', 'additive', and 'minimum'}.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    custom_interpolation_kernel_2d : np.ndarray
        Convolution kernel used with the 'custom' interpolation mode in 2D rendering.
    depiction : str
        3D Depiction mode. Must be one of {'volume', 'plane'}.
        The default value is 'volume'.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.
    metadata : dict
        Layer metadata.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape
        then it will be taken to be multiscale. The first image in the list
        should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    name : str
        Name of the layer. If not provided then will be guessed using heuristics.
    ndim : int
        Number of dimensions in the data.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    plane : dict or SlicingPlane
        Properties defining plane rendering in 3D. Properties are defined in
        data coordinates. Valid dictionary keys are
        {'position', 'normal', 'thickness', and 'enabled'}.
    projection_mode : str
        How data outside the viewed dimensions but inside the thick Dims slice will
        be projected onto the viewed dimensions. Must fit to cls._projectionclass.
    rendering : str
        Rendering mode used by vispy. Must be one of our supported
        modes.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    scale : tuple of float
        Scale factors for the layer.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    translate : tuple of float
        Translation values for the layer.
    units : tuple of str or pint.Unit, optional
        Units of the layer data in world coordinates.
        If not provided, the default units are assumed to be pixels.
    visible : bool
        Whether the layer visual is currently being displayed.


    Attributes
    ----------
    data : array or list of array
        Image data. Can be N dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a list
        and arrays are decreasing in shape then the data is treated as a
        multiscale image. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    axis_labels : tuple of str
        Dimension names of the layer data.
    custom_interpolation_kernel_2d : np.ndarray
        Convolution kernel used with the 'custom' interpolation mode in 2D rendering.
    depiction : str
        3D Depiction mode used by vispy. Must be one of our supported modes.
    experimental_clipping_planes : ClippingPlaneList
        Clipping planes defined in data coordinates, used to clip the volume.
    metadata : dict
        Image metadata.
    mode : str
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In TRANSFORM mode the image can be transformed interactively.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. The first image in the
        list should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    plane : SlicingPlane or dict
        Properties defining plane rendering in 3D. Valid dictionary keys are
        {'position', 'normal', 'thickness'}.
    rendering : str
        Rendering mode used by vispy. Must be one of our supported
        modes.
    units: tuple of pint.Unit
        Units of the layer data in world coordinates.

    Notes
    -----
    _data_view : array (N, M), (N, M, 3), or (N, M, 4)
        Image data for the currently viewed slice. Must be 2D image data, but
        can be multidimensional for RGB or RGBA images if multidimensional is
        `True`.
    """

    _colormaps = AVAILABLE_COLORMAPS
    _interpolation2d: Interpolation
    _interpolation3d: Interpolation
    _level_materializer: Callable[[int], np.ndarray] | None
    _slicing_state: ScalarFieldSlicingState

    def __init__(
        self,
        data,
        *,
        affine=None,
        axis_labels=None,
        blending='translucent',
        cache=True,
        custom_interpolation_kernel_2d=None,
        depiction='volume',
        experimental_clipping_planes=None,
        metadata=None,
        multiscale=None,
        name=None,
        ndim=None,
        opacity=1.0,
        plane=None,
        projection_mode='none',
        rendering='mip',
        rotate=None,
        scale=None,
        shear=None,
        translate=None,
        units=None,
        visible=True,
    ):
        if name is None and data is not None:
            name = magic_name(data)

        if isinstance(data, types.GeneratorType):
            data = list(data)

        if getattr(data, 'ndim', 2) < 2:
            raise ValueError(
                trans._('Image data must have at least 2 dimensions.')
            )

        # Determine if data is a multiscale
        self._data_raw = data
        if multiscale is None:
            multiscale, data = guess_multiscale(data)
        elif multiscale and not isinstance(data, MultiScaleData):
            data = MultiScaleData(data)

        # Determine dimensionality of the data
        if ndim is None:
            ndim = len(data.shape)
        self._data = data

        super().__init__(
            data,
            ndim,
            affine=affine,
            axis_labels=axis_labels,
            blending=blending,
            cache=cache,
            experimental_clipping_planes=experimental_clipping_planes,
            metadata=metadata,
            multiscale=multiscale,
            name=name,
            opacity=opacity,
            projection_mode=projection_mode,
            scale=scale,
            shear=shear,
            rotate=rotate,
            translate=translate,
            units=units,
            visible=visible,
        )

        self.events.add(
            attenuation=Event,
            custom_interpolation_kernel_2d=Event,
            depiction=Event,
            locked_data_level=Event,
            interpolation=WarningEmitter(
                trans._(
                    "'layer.events.interpolation' is deprecated please use `interpolation2d` and `interpolation3d`",
                    deferred=True,
                ),
                type_name='select',
            ),
            interpolation2d=Event,
            interpolation3d=Event,
            iso_threshold=Event,
            plane=Event,
            rendering=Event,
        )

        self._array_like = True

        # User-override for multiscale data level.
        # When not None, _update_draw will use this level instead of
        # automatically selecting one based on the viewport / 3D mode.
        self._locked_data_level: int | None = None

        # Viewport bbox from the most recent draw, as
        # (displayed_axes, data_bbox_int); see _update_level_and_corners.
        self._last_data_bbox: tuple | None = None

        # Experimental: maximum per-axis extent (in data pixels) of the
        # region sliced for a locked multiscale level in 3D. When set,
        # locking a level larger than this renders a sub-volume tile of
        # this size centered on the current view (tracked across camera
        # moves) instead of the full level — making levels that exceed
        # GL texture limits usable. ``None`` (default) keeps the previous
        # behavior of always slicing the full locked level.
        self._max_tile_extent_3d: int | None = None
        self._tile_max_bytes_3d: int | None = None

        # Set data
        self._data = data
        if isinstance(data, MultiScaleData):
            self._data_level = len(data) - 1
        else:
            self._data_level = 0
        displayed_axes = self._slice_input.displayed
        self.corner_pixels[1][displayed_axes] = (
            np.array(self.level_shapes)[self._data_level][displayed_axes] - 1
        )
        self._reset_thumbnail_level_data()

        self._plane = SlicingPlane(thickness=1)
        # Whether to calculate clims on the next set_view_slice
        self._should_calc_clims = False
        # using self.colormap = colormap uses the setter in *derived* classes,
        # where the intention here is to use the base setter, so we use the
        # _set_colormap method. This is important for Labels layers, because
        # we don't want to use get_color before set_view_slice has been
        # triggered (self.refresh(), below).
        self.rendering = rendering
        self.depiction = depiction
        if plane is not None:
            self.plane = plane
        connect_no_arg(self.plane.events, self.events, 'plane')
        self.custom_interpolation_kernel_2d = custom_interpolation_kernel_2d

    def _post_init(self):
        # Trigger generation of view slice and thumbnail
        self.refresh()

    def _slice_dtype(self):
        """Return the dtype of the slice view.

        Overridden in Labels subclass to properly handle the
        32 and 64 bits dtypes.
        """
        return self.dtype

    @property
    def _data_view(self) -> np.ndarray:
        """Viewable image for the current slice. (compatibility)"""
        return self._slice.image.view

    @property
    def _slice(self) -> _ScalarFieldSliceResponse:
        return self._slicing_state._slice

    @property
    def dtype(self):
        return normalize_dtype(self._data.dtype)

    @property
    def data_raw(
        self,
    ) -> LayerDataProtocol | Sequence[LayerDataProtocol]:
        """Data, exactly as provided by the user."""
        return self._data_raw

    @property
    def data(self) -> LayerDataProtocol | MultiScaleData:
        """Data, possibly in multiscale wrapper. Obeys LayerDataProtocol."""
        return self._data

    @data.setter
    def data(self, data: LayerDataProtocol | MultiScaleData) -> None:
        self._data_raw = data
        # note, we don't support changing from/to multiscale after construction
        self._data = MultiScaleData(data) if self.multiscale else data  # type: ignore[arg-type]
        self._reset_data_level()
        self._reset_thumbnail_level_data()
        self._update_dims()
        self.events.data(value=self.data)
        self._reset_editable()

    def _get_ndim(self) -> int:
        """Determine number of dimensions of the layer."""
        return len(self.level_shapes[0])

    @property
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        shape = self.level_shapes[0]
        return np.vstack([np.zeros(len(shape)), shape - 1])

    @property
    def _extent_data_augmented(self) -> np.ndarray:
        extent = self._extent_data
        return extent + [[-0.5], [+0.5]]

    @property
    def _extent_level_data(self) -> np.ndarray:
        """Extent of layer, accounting for current multiscale level, in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        shape = self.level_shapes[self.data_level]
        return np.vstack([np.zeros(len(shape)), shape - 1])

    @property
    def _extent_level_data_augmented(self) -> np.ndarray:
        extent = self._extent_level_data
        return extent + [[-0.5], [+0.5]]

    @property
    def data_level(self) -> int:
        """int: Current level of multiscale, or 0 if image."""
        return self._data_level

    @data_level.setter
    def data_level(self, level: int) -> None:
        if self._data_level == level:
            return
        self._data_level = level
        self.refresh(extent=False)

    @property
    def locked_data_level(self) -> int | None:
        """int or None: Locked multiscale resolution level.

        When set to an integer, forces rendering at the given multiscale
        level instead of automatic level selection based on the viewport.
        Set to ``None`` to restore automatic behaviour.


        .. versionadded:: 0.7.1
        """
        return self._locked_data_level

    @locked_data_level.setter
    def locked_data_level(self, level: int | None) -> None:
        if level is not None:
            n_levels = len(self.level_shapes)
            if level < 0 or level >= n_levels:
                return
        old_level = self._data_level
        self._locked_data_level = level
        if level is not None:
            displayed_axes = self._slice_input.displayed
            # Size any 3D sub-volume tile to the viewport instead of the
            # full memory budget — slicing a budget-sized cube here would
            # stall the UI before the next draw refines it. Use the
            # viewport bbox cached from the last draw when available,
            # falling back to the previous level's corner pixels.
            data_bbox_int = None
            if self._last_data_bbox is not None and self._last_data_bbox[
                0
            ] == tuple(displayed_axes):
                data_bbox_int = self._last_data_bbox[1]
            else:
                data_bbox_int = (
                    self.corner_pixels[:, displayed_axes]
                    * np.take(
                        np.asarray(self.downsample_factors[old_level]),
                        displayed_axes,
                    )
                ).astype(int)
            self.corner_pixels = self._corners_for_locked_level(
                level, displayed_axes, data_bbox_int
            )
            self._data_level = level
        else:
            self._reset_data_level()
        self.refresh(extent=False)
        self.events.locked_data_level()

    def _reset_data_level(self) -> None:
        """Reset ``_locked_data_level`` and ``_data_level`` for new data.

        Called from the ``data`` setter of subclasses when the underlying
        array is replaced.  Uses the coarsest level for multiscale data
        (matching ``__init__`` behaviour) and 0 for single-scale data.
        """
        self._locked_data_level = None
        if isinstance(self._data, MultiScaleData):
            self._data_level = len(self._data) - 1
        else:
            self._data_level = 0

    def _update_level_and_corners(
        self, data_bbox_int, shape_threshold, displayed_axes
    ):
        """Update the data level and corner pixels for the current viewport.

        For multiscale layers, selects the appropriate resolution level
        (locked, 2D auto, or 3D coarsest), computes corner pixels for that
        level, and refreshes the layer when the level or visible region
        changes. For non-multiscale data, delegates to the base implementation.
        """
        if not self.multiscale:
            super()._update_level_and_corners(
                data_bbox_int, shape_threshold, displayed_axes
            )
            return

        # remember the viewport bbox (level-0 data coords) so that other
        # corner computations (e.g. the locked_data_level setter, which
        # runs outside a draw) can size 3D sub-volume tiles to the view
        self._last_data_bbox = (tuple(displayed_axes), data_bbox_int)

        if self._locked_data_level is not None:
            # User has explicitly locked the data level; skip automatic
            # level selection and use the full extent of that level (or a
            # view-centered sub-volume tile in 3D when the level exceeds
            # _max_tile_extent_3d).
            locked = self._locked_data_level
            old_level = self._data_level
            self._data_level = locked
            corners = self._corners_for_locked_level(
                locked, displayed_axes, data_bbox_int
            )
            level_changed = old_level != locked
            if level_changed or self._locked_tile_moved(
                corners, displayed_axes
            ):
                self.corner_pixels = corners
                self.refresh(extent=False, thumbnail=False)
        elif self._slice_input.ndisplay == 2:
            level, scaled_corners = compute_multiscale_level_and_corners(
                data_bbox_int,
                shape_threshold,
                self.downsample_factors[:, displayed_axes],
            )
            margin = getattr(self, '_render_margin_2d', 1.0)
            if margin > 1.0:
                # Render a margin around the viewport so pans and
                # zoom-outs stay inside already-sliced content instead
                # of exposing unrendered void. A factor of 2 covers
                # zoom-out exactly to the next level switch in a
                # factor-2 pyramid. Set by progressive loading.
                pad = (
                    (scaled_corners[1] - scaled_corners[0])
                    * (margin - 1.0)
                    / 2.0
                )
                scaled_corners = np.stack(
                    [scaled_corners[0] - pad, scaled_corners[1] + pad],
                )
            corners = np.zeros((2, self.ndim), dtype=int)
            max_coords = np.take(self.data[level].shape, displayed_axes) - 1
            corners[:, displayed_axes] = np.clip(scaled_corners, 0, max_coords)
            display_shape = tuple(
                corners[1, displayed_axes] - corners[0, displayed_axes]
            )
            if any(s == 0 for s in display_shape):
                return
            # Only update when level changes or
            # when new view is outside current corner_pixels
            if (
                self.data_level != level
                or np.any(
                    corners[0, displayed_axes]
                    < self.corner_pixels[0, displayed_axes]
                )
                or np.any(
                    corners[1, displayed_axes]
                    > self.corner_pixels[1, displayed_axes]
                )
            ):
                self._data_level = level
                self.corner_pixels = corners
                self.refresh(extent=False, thumbnail=False)
        else:
            # 3D: use the coarsest level, full extent
            new_level = len(self.level_shapes) - 1
            level_changed = self._data_level != new_level
            self._data_level = new_level
            corners = np.zeros((2, self.ndim), dtype=int)
            corners[1, displayed_axes] = (
                np.take(self.data[new_level].shape, displayed_axes) - 1
            )
            self.corner_pixels = corners
            if level_changed:
                self.refresh(extent=False, thumbnail=False)

    def _corners_for_locked_level(
        self,
        level: int,
        displayed_axes: list[int],
        data_bbox_int: np.ndarray | None = None,
    ) -> np.ndarray:
        """Corner pixels to render for a locked multiscale level.

        Normally the full extent of the level. In 3D, when
        ``_max_tile_extent_3d`` is set and the level is larger than that
        extent along a displayed axis, a sub-volume tile of at most that
        extent is returned instead, centered on ``data_bbox_int`` (the
        current view in level-0 data coordinates) or the middle of the
        level.
        """
        shape_at_level = np.take(
            np.asarray(self.level_shapes[level]), displayed_axes
        )
        corners = np.zeros((2, self.ndim), dtype=int)
        corners[1, displayed_axes] = shape_at_level - 1

        extent_cap = self._max_tile_extent_3d
        if self._slice_input.ndisplay != 3 or extent_cap is None:
            return corners

        downsample = np.take(
            np.asarray(self.downsample_factors[level]), displayed_axes
        )
        # Compute an anisotropic tile extent that fits the byte budget.
        # Axes already smaller than the cap keep their full size; the
        # budget is distributed to larger axes so anisotropic data
        # (e.g. Z=42, Y=304, X=657) shows more of the volume.
        tile_bytes = self._tile_max_bytes_3d
        if tile_bytes is not None:
            itemsize = max(int(self.dtype.itemsize), 1)
            max_elements = max(tile_bytes // itemsize, 1)
            # Start from the full level shape (capped per-axis by the GL
            # texture limit), then shrink proportionally to fit the byte
            # budget. Small axes keep their full size while larger axes
            # shrink — so anisotropic data shows much more of the volume
            # than a uniform cube cap would.
            try:
                from napari._vispy.utils.gl import get_max_texture_sizes
                _, gl_max = get_max_texture_sizes()
            except Exception:  # noqa: BLE001
                gl_max = extent_cap
            if gl_max is None:
                gl_max = extent_cap
            tile_extent = np.minimum(shape_at_level, gl_max).astype(np.int64)
            for _ in range(len(tile_extent)):
                vol = int(np.prod(tile_extent))
                if vol <= max_elements:
                    break
                over = np.where(tile_extent > _MIN_TILE_EXTENT_3D)[0]
                if len(over) == 0:
                    break
                ratio = (max_elements / vol) ** (1.0 / len(over))
                for ax in over:
                    tile_extent[ax] = max(int(tile_extent[ax] * ratio),
                                          _MIN_TILE_EXTENT_3D)
        else:
            tile_extent = np.minimum(shape_at_level, extent_cap)
        if data_bbox_int is not None and np.all(np.isfinite(data_bbox_int)):
            bbox = np.asarray(data_bbox_int, dtype=float) / downsample
            center = bbox.mean(axis=0)
            # Bound the tile by the visible extent so deep zooms slice
            # (and fetch) only what is on screen. The canvas-plane bbox is
            # degenerate along the view axis, so give every axis at least
            # the largest on-screen extent (a view-sized cube). A margin
            # (set by progressive loading) adds pan slack so small camera
            # translations stay inside the tile instead of re-slicing.
            view_extent = bbox[1] - bbox[0]
            view_extent = np.maximum(view_extent, view_extent.max())
            view_extent = view_extent * getattr(self, '_tile_margin_3d', 1.0)
            view_extent = np.ceil(view_extent).astype(np.int64)
            if np.all(view_extent > 0):
                tile_extent = np.minimum(
                    tile_extent,
                    np.maximum(view_extent, _MIN_TILE_EXTENT_3D),
                )
        else:
            center = shape_at_level / 2
        # quantize so tile shapes repeat across camera poses (texture
        # reuse); the stable caps (extent_cap, level shape) still apply
        quantum = _TILE_EXTENT_QUANTUM_3D
        tile_extent = np.asarray(tile_extent, dtype=np.int64)
        tile_extent = np.minimum(
            -(-tile_extent // quantum) * quantum,
            shape_at_level,
        )
        center = np.clip(center, 0, shape_at_level - 1)
        if np.all(tile_extent >= shape_at_level):
            return corners
        low = np.clip(
            (center - tile_extent / 2).astype(int),
            0,
            shape_at_level - tile_extent,
        )
        high = low + tile_extent
        corners[0, displayed_axes] = low
        corners[1, displayed_axes] = high - 1
        return corners

    def _locked_tile_moved(
        self, corners: np.ndarray, displayed_axes: list[int]
    ) -> bool:
        """Whether new locked-level corners warrant a re-slice.

        Uses hysteresis of a quarter of the tile extent so that small
        camera movements do not continuously re-slice a sub-volume tile.
        """
        current = self.corner_pixels
        if current.shape != corners.shape:
            return True
        extent = (corners[1] - corners[0])[displayed_axes] + 1
        current_extent = (current[1] - current[0])[displayed_axes] + 1
        if not np.array_equal(extent, current_extent):
            return True
        new_center = corners.mean(axis=0)[displayed_axes]
        current_center = current.mean(axis=0)[displayed_axes]
        return bool(np.any(np.abs(new_center - current_center) > extent / 4))

    def _reset_thumbnail_level_data(self) -> None:
        """Set ``_thumbnail_level`` and ``_level_materializer`` for the current data.

        Called once during ``__init__`` and again whenever ``data`` is replaced.
        Single-scale and 3D multiscale layers set ``_level_materializer`` to
        ``None``; only 2D multiscale layers cache the thumbnail level.
        """
        data = self._data
        if isinstance(data, MultiScaleData):
            self._thumbnail_level = len(data) - 1
            self._level_materializer = (
                _make_level_materializer(data)
                if self._get_ndim() == 2
                else None
            )
        else:
            self._thumbnail_level = 0
            self._level_materializer = None

    def _get_level_shapes(self) -> Sequence[tuple[int, ...]]:
        data = self.data
        if isinstance(data, MultiScaleData):
            return data.shapes
        return [self.data.shape]

    @property
    def level_shapes(self) -> np.ndarray:
        """array: Shapes of each level of the multiscale or just of image."""
        return np.array(self._get_level_shapes())

    @property
    def downsample_factors(self) -> np.ndarray:
        """list: Downsample factors for each level of the multiscale."""
        return np.divide(self.level_shapes[0], self.level_shapes)

    @property
    def depiction(self):
        """The current 3D depiction mode.

        Selects a preset depiction mode in vispy
            * volume: images are rendered as 3D volumes.
            * plane: images are rendered as 2D planes embedded in 3D.
                plane position, normal, and thickness are attributes of
                layer.plane which can be modified directly.
        """
        return str(self._depiction)

    @depiction.setter
    def depiction(self, depiction: str | VolumeDepiction) -> None:
        """Set the current 3D depiction mode."""
        self._depiction = VolumeDepiction(depiction)
        self._update_plane_callbacks()
        self.events.depiction()

    def _reset_plane_parameters(self):
        """Set plane attributes to something valid."""
        self.plane.position = np.array(self.data.shape) / 2
        self.plane.normal = (1, 0, 0)

    def _update_plane_callbacks(self):
        """Set plane callbacks depending on depiction mode."""
        plane_drag_callback_connected = (
            plane_drag_callback in self.mouse_drag_callbacks
        )
        double_click_callback_connected = (
            plane_double_click_callback in self.mouse_double_click_callbacks
        )
        if self.depiction == VolumeDepiction.VOLUME:
            if plane_drag_callback_connected:
                self.mouse_drag_callbacks.remove(plane_drag_callback)
            if double_click_callback_connected:
                self.mouse_double_click_callbacks.remove(
                    plane_double_click_callback
                )
        elif self.depiction == VolumeDepiction.PLANE:
            if not plane_drag_callback_connected:
                self.mouse_drag_callbacks.append(plane_drag_callback)
            if not double_click_callback_connected:
                self.mouse_double_click_callbacks.append(
                    plane_double_click_callback
                )

    @property
    def plane(self):
        return self._plane

    @plane.setter
    def plane(self, value: dict | SlicingPlane) -> None:
        self._plane.update(value)
        self.events.plane()

    @property
    def custom_interpolation_kernel_2d(self):
        return self._custom_interpolation_kernel_2d

    @custom_interpolation_kernel_2d.setter
    def custom_interpolation_kernel_2d(self, value):
        if value is None:
            value = [[1]]
        self._custom_interpolation_kernel_2d = np.array(value, np.float32)
        self.events.custom_interpolation_kernel_2d()

    @abstractmethod
    def _raw_to_displayed(self, raw: np.ndarray) -> np.ndarray:
        """Determine displayed image from raw image.

        For normal image layers, just return the actual image.

        Parameters
        ----------
        raw : array
            Raw array.

        Returns
        -------
        image : array
            Displayed array.
        """
        raise NotImplementedError

    def _set_view_slice(self):
        raise NotImplementedError

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
        if self.multiscale:
            # for multiscale data map the coordinate from the data back to
            # the tile
            coord = self._transforms['tile2data'].inverse(position)
        else:
            coord = position

        coord = np.round(coord).astype(int)

        raw = self._slice.image.raw
        shape = (
            raw.shape[:-1] if self.ndim != len(self._data.shape) else raw.shape
        )

        if self.ndim < len(coord):
            # handle 3D views of 2D data by omitting extra coordinate
            offset = len(coord) - len(shape)
            coord = coord[[d + offset for d in self._slice_input.displayed]]
        else:
            coord = coord[self._slice_input.displayed]

        if all(0 <= c < s for c, s in zip(coord, shape, strict=False)):
            value = raw[tuple(coord)]
        else:
            value = None

        if self.multiscale:
            value = (self.data_level, value)

        return value

    def _get_value_ray(
        self,
        start_point: np.ndarray | None,
        end_point: np.ndarray | None,
        dims_displayed: list[int],
    ) -> int | None:
        """Get the first non-background value encountered along a ray.

        Parameters
        ----------
        start_point : np.ndarray
            (n,) array containing the start point of the ray in data coordinates.
        end_point : np.ndarray
            (n,) array containing the end point of the ray in data coordinates.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the viewer.

        Returns
        -------
        value : Optional[int]
            The first non-background value encountered along the ray. If none
            was encountered or the viewer is in 2D mode, returns None.
        """
        if start_point is None or end_point is None:
            return None
        if len(dims_displayed) == 3:
            # only use get_value_ray on 3D for now
            # we use dims_displayed because the image slice
            # has its dimensions in the same order as the vispy
            # Volume.
            #
            # Grab the slice data first, then derive the downsample
            # factor from its actual shape so that coordinates and
            # data are always consistent (data_level and the slice
            # can be temporarily out of sync).
            im_slice = self._slice.image.raw
            slice_shape = np.array(im_slice.shape)
            level0_shape = np.array(self.level_shapes[0])
            ds = level0_shape[dims_displayed] / slice_shape
            start_point = start_point[dims_displayed] / ds
            end_point = end_point[dims_displayed] / ds
            start_point = cast(np.ndarray, start_point)
            end_point = cast(np.ndarray, end_point)
            sample_ray = end_point - start_point
            length_sample_vector = np.linalg.norm(sample_ray)
            n_points = int(2 * length_sample_vector)
            sample_points = np.linspace(
                start_point, end_point, n_points, endpoint=True
            )
            # Build the bounding box from the actual slice shape
            bounding_box = np.zeros((len(dims_displayed), 2))
            bounding_box[:, 1] = slice_shape

            clamped = clamp_point_to_bounding_box(
                sample_points,
                bounding_box,
            ).astype(int)
            values = im_slice[tuple(clamped.T)]
            return self._calculate_value_from_ray(values)

        return None

    @abstractmethod
    def _calculate_value_from_ray(self, values):
        raise NotImplementedError

    def _get_value_3d(
        self,
        start_point: np.ndarray | None,
        end_point: np.ndarray | None,
        dims_displayed: list[int],
    ) -> int | None | tuple[int, int | None]:
        """Get the first non-background value encountered along a ray.

        Parameters
        ----------
        start_point : np.ndarray
            (n,) array containing the start point of the ray in data coordinates.
        end_point : np.ndarray
            (n,) array containing the end point of the ray in data coordinates.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the viewer.

        Returns
        -------
        value : int or tuple
            The first non-zero value encountered along the ray. If a
            non-zero value is not encountered, returns None.
            If multiscale is True, returns a tuple of (data_level, value).
        """
        value = self._get_value_ray(
            start_point=start_point,
            end_point=end_point,
            dims_displayed=dims_displayed,
        )

        if self.multiscale and value is not None:
            return self.data_level, value

        return value

    def _get_offset_data_position(self, position: npt.NDArray) -> npt.NDArray:
        """Adjust position for offset between viewer and data coordinates.

        VisPy considers the coordinate system origin to be the canvas corner,
        while napari considers the origin to be the **center** of the corner
        pixel. To get the correct value under the mouse cursor, we need to
        shift the position by 0.5 pixels on each axis.
        """
        return position + 0.5

    def _display_bounding_box_at_level(
        self, dims_displayed: list[int], data_level: int
    ) -> npt.NDArray:
        """An axis aligned (ndisplay, 2) bounding box around the data at a given level"""
        shape = self.level_shapes[data_level]
        extent_at_level = np.vstack([np.zeros(len(shape)), shape - 1])
        return extent_at_level[:, dims_displayed].T

    def _display_bounding_box_augmented_data_level(
        self, dims_displayed: list[int]
    ) -> npt.NDArray:
        """An augmented, axis-aligned (ndisplay, 2) bounding box.
        If the layer is multiscale layer, then returns the
        bounding box of the data at the current level
        """
        return self._extent_level_data_augmented[:, dims_displayed].T

    def _get_layer_slicing_state(
        self, data: LayerDataType, cache: bool
    ) -> ScalarFieldSlicingState:
        return ScalarFieldSlicingState(layer=self, data=data, cache=cache)


class ScalarFieldSlicingState(_LayerSlicingState):
    layer: ScalarFieldBase
    _slice_request_class = _ScalarFieldSliceRequest

    def __init__(
        self, layer: ScalarFieldBase, data: LayerDataType, cache: bool
    ):
        super().__init__(layer, data, cache)
        self.transforms = Affine(
            np.ones(self.ndim), np.zeros(self.ndim), name='tile2data'
        )
        self._slice = _ScalarFieldSliceResponse.make_empty(
            slice_input=self._slice_input,
            rgb=len(self.layer.data.shape) != self.ndim,
            dtype=self.layer._slice_dtype(),
        )

    def _set_view_slice(self):
        request = self._make_slice_request_internal(
            slice_input=self._slice_input,
            data_slice=self.data_slice,
            dask_indexer=nullcontext,
        )
        response = request()
        self._update_slice_response(response)

    def set_slice_input(self, slice_input: _SliceInput, force: bool) -> bool:
        changed = super().set_slice_input(slice_input, force)
        # When the layer is invisible the parent skips set_view_slice to avoid
        # eagerly fetching data on add, so the cached _slice still has the old
        # slice_input. Its placeholder image then has the wrong rank for the
        # new ndisplay (e.g. (1, 1) when ndisplay flipped 2 -> 3). vispy reads
        # layer._data_view regardless of layer.visible and hands it to
        # node.set_data, which rejects the wrong-rank array. Refresh just the
        # placeholder so its shape matches; this stays cheap and never touches
        # the underlying data.
        if changed and not self.layer.visible:
            self._slice = _ScalarFieldSliceResponse.make_empty(
                slice_input=self._slice_input,
                rgb=len(self.layer.data.shape) != self.ndim,
                dtype=self.layer._slice_dtype(),
            )
        return changed

    def _make_slice_request(self, dims: Dims) -> _ScalarFieldSliceRequest:
        """Make an image slice request based on the given dims and this image."""
        slice_input = self.make_slice_input(dims)
        # For the existing sync slicing, indices is passed through
        # to avoid some performance issues related to the evaluation of the
        # data-to-world transform and its inverse. Async slicing currently
        # absorbs these performance issues here, but we can likely improve
        # things either by caching the world-to-data transform on the layer
        # or by lazily evaluating it in the slice task itself.
        data_slice = self._slice_indices(slice_input, dims)
        return self._make_slice_request_internal(
            slice_input=slice_input,
            data_slice=data_slice,
            dask_indexer=self.dask_optimized_slicing,
        )

    def _make_slice_request_internal(
        self,
        *,
        slice_input: _SliceInput,
        data_slice: _ThickNDSlice,
        dask_indexer: DaskIndexer,
    ) -> _ScalarFieldSliceRequest:
        """Needed to support old-style sync slicing through _slice_dims and
        _set_view_slice.

        This is temporary scaffolding that should go away once we have completed
        the async slicing project: https://github.com/napari/napari/issues/4795
        """
        data = self.layer.data
        if self.layer.multiscale:
            locked = self.layer._locked_data_level
            if locked is not None:
                data_level = locked
            elif slice_input.ndisplay == 3:
                data_level = len(data) - 1  # type: ignore[arg-type]
            else:
                data_level = self.layer.data_level
        else:
            data_level = 0

        thumbnail_level = self.layer._thumbnail_level
        data_at_thumbnail_level: (
            LayerDataProtocol | MultiScaleData | np.ndarray
        )
        if self.layer._level_materializer:
            data_at_thumbnail_level = self.layer._level_materializer(
                thumbnail_level
            )
        elif self.layer.multiscale:
            data_at_thumbnail_level = data[thumbnail_level]
        else:
            data_at_thumbnail_level = data

        data_at_data_level: LayerDataProtocol | MultiScaleData | np.ndarray
        if not self.layer.multiscale:
            data_at_data_level = data
        elif data_level == thumbnail_level:
            data_at_data_level = data_at_thumbnail_level
        else:
            data_at_data_level = data[data_level]

        return self._slice_request_class(
            slice_input=slice_input,
            data_at_data_level=data_at_data_level,
            data_at_thumbnail_level=data_at_thumbnail_level,
            dtype=self.layer.dtype,
            dask_indexer=dask_indexer,
            data_slice=data_slice,
            projection_mode=self.layer.projection_mode,
            multiscale=self.layer.multiscale,
            corner_pixels=self.layer.corner_pixels,
            rgb=len(self.layer.data.shape) != self.ndim,
            data_level=data_level,
            thumbnail_level=thumbnail_level,
            level_shapes=self.layer.level_shapes,
            downsample_factors=self.layer.downsample_factors,
        )

    def _update_slice_response(
        self, response: _ScalarFieldSliceResponse
    ) -> None:
        """Update the slice output state currently on the layer. Currently used
        for both sync and async slicing.
        """
        response = response.to_displayed(self.layer._raw_to_displayed)
        # We call to_displayed here to ensure that if the contrast limits
        # are outside the range of supported by vispy, then data view is
        # rescaled to fit within the range.
        self._slice_input = response.slice_input
        # this is the temporary patch
        self.layer._transforms[0] = response.tile_to_data
        #
        self.transforms = response.tile_to_data
        self._slice = response
