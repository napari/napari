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
    data: Any = field(repr=False)
    thumbnail: Optional[Any] = field(repr=False)
    tile_to_data: Optional[Affine] = field(repr=False)


@dataclass(frozen=True)
class _ImageSliceRequest:
    """Represents a single image slice request.

    This should be treated a deeply immutable structure, even though some
    fields can be modified in place.

    In general, the execute method may take a long time to run, so you may
    want to run it once on a worker thread.
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
        if self.multiscale:
            return self._execute_multi_scale()
        return self._execute_single_scale()

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
