import warnings
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Optional, Tuple, Union

import numpy as np

from napari.layers.utils._slice_input import _SliceInput
from napari.utils.transforms import Affine
from napari.utils.translations import trans


@dataclass(frozen=True)
class _ImageSliceResponse:
    """Contains all the output data of slicing an image layer.

    Attributes
    ----------
    data : Any
        The sliced image data.
        In general, if you need this to be a `numpy.ndarray` you should call `np.asarray`.
        Though if the corresponding request was not lazy, this is likely a `numpy.ndarray`.
    thumbnail: Optional[Any]
        The thumbnail image data, which may be a different resolution to the sliced image data
        for multi-scale images.
        For single-scale images, this will be `None`, which indicates that the thumbnail data
        is the same as the sliced image data.
    tile_to_data: Optional[Affine]
        The affine transform from the sliced data to the full data at the highest resolution.
        For single-scale images, this will be `None`.
    """

    data: Any = field(repr=False)
    thumbnail: Optional[Any] = field(repr=False)
    tile_to_data: Optional[Affine] = field(repr=False)


@dataclass(frozen=True)
class _ImageSliceRequest:
    """Contains all the input data needed to slice an image layer.

    This should be treated a deeply immutable structure, even though some
    fields can be modified in place.

    In general, the execute method may take a long time to run, so you may
    want to run it once on a worker thread.

    Attributes
    ----------
    dims : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    data : Any
        The layer's data field, which is the main input to slicing.
    data_to_world: Affine
        The affine transform from the data at the highest resolution to the
        layer's world coordinate system.
    lazy : bool
        If True, do not materialize the data with `np.asarray` during execution.
        Otherwise, False. This should be True for the experimental async code
        (as the load occurs on a separate thread) but False for the new async
        where `execute` is expected to be run on a separate thread.
    others
        See the corresponding attributes in `Layer` and `Image`.
    """

    dims: _SliceInput
    data: Any = field(repr=False)
    data_to_world: Affine = field(repr=False)
    multiscale: bool = field(repr=False)
    corner_pixels: np.ndarray
    rgb: bool = field(repr=False)
    data_level: int = field(repr=False)
    thumbnail_level: int = field(repr=False)
    level_shapes: np.ndarray = field(repr=False)
    downsample_factors: np.ndarray = field(repr=False)
    lazy: bool = field(default=False, repr=False)

    @cached_property
    def slice_indices(self) -> Tuple[Union[int, float, slice], ...]:
        if len(self.dims.not_displayed) == 0:
            return (slice(None),) * self.dims.ndim
        return self.dims.data_indices(self.data_to_world.inverse)

    def execute(self) -> _ImageSliceResponse:
        return (
            self._execute_multi_scale()
            if self.multiscale
            else self._execute_single_scale()
        )

    def _execute_single_scale(self) -> _ImageSliceResponse:
        image = self.data[self.slice_indices]
        if not self.lazy:
            image = np.asarray(image)
        return _ImageSliceResponse(
            data=image,
            thumbnail=None,
            tile_to_data=None,
        )

    def _execute_multi_scale(self) -> _ImageSliceResponse:
        if self.dims.ndisplay == 3:
            warnings.warn(
                trans._(
                    'Multiscale rendering is only supported in 2D. In 3D, only the lowest resolution scale is displayed',
                    deferred=True,
                ),
                category=UserWarning,
            )
            level = len(self.data) - 1
        else:
            level = self.data_level

        indices = self._slice_indices_at_level(level)

        scale = np.ones(self.dims.ndim)
        for d in self.dims.displayed:
            scale[d] = self.downsample_factors[level][d]

        # This only needs to be a ScaleTranslate but different types
        # of transforms in a chain don't play nicely together right now.
        tile_to_data = Affine(name='tile2data', scale=scale)
        if self.dims.ndisplay == 2:
            for d in self.dims.displayed:
                indices[d] = slice(
                    self.corner_pixels[0, d],
                    self.corner_pixels[1, d],
                    1,
                )
            # TODO: why do we only do this for 2D display?
            # I guess we only support multiscale in 2D anyway, but then why is scale
            # not in here too?
            tile_to_data.translate = self.corner_pixels[0] * tile_to_data.scale

        thumbnail_indices = self._slice_indices_at_level(self.thumbnail_level)

        image = self.data[level][tuple(indices)]
        thumbnail = self.data[self.thumbnail_level][tuple(thumbnail_indices)]

        if not self.lazy:
            image = np.asarray(image)
            thumbnail = np.asarray(thumbnail)

        return _ImageSliceResponse(
            data=image,
            thumbnail=thumbnail,
            tile_to_data=tile_to_data,
        )

    def _slice_indices_at_level(
        self, level: int
    ) -> Tuple[Union[int, float, slice], ...]:
        indices = np.array(self.slice_indices)
        axes = self.dims.not_displayed
        ds_indices = indices[axes] / self.downsample_factors[level][axes]
        ds_indices = np.round(ds_indices.astype(float)).astype(int)
        ds_indices = np.clip(ds_indices, 0, self.level_shapes[level][axes] - 1)
        indices[axes] = ds_indices
        return indices
