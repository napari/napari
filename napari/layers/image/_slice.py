import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union

import numpy as np

from napari.layers.utils._slice_input import _SliceInput
from napari.utils._dask_utils import DaskIndexer
from napari.utils.transforms import Affine
from napari.utils.translations import trans


@dataclass(frozen=True)
class _ImageSliceResponse:
    """Contains all the output data of slicing an image layer.

    Attributes
    ----------
    data : array like
        The sliced image data.
        In general, if you need this to be a `numpy.ndarray` you should call `np.asarray`.
        Though if the corresponding request was not lazy, this is likely a `numpy.ndarray`.
    thumbnail: array like or none
        The thumbnail image data, which may be a different resolution to the sliced image data
        for multi-scale images.
        For single-scale images, this will be `None`, which indicates that the thumbnail data
        is the same as the sliced image data.
    tile_to_data: Affine
        The affine transform from the sliced data to the full data at the highest resolution.
        For single-scale images, this will be the identity matrix.
    dims : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    indices : tuple of ints or slices
        The slice indices in the layer's data space.
    """

    data: Any = field(repr=False)
    thumbnail: Optional[Any] = field(repr=False)
    tile_to_data: Affine = field(repr=False)
    dims: _SliceInput
    indices: Tuple[Union[int, slice], ...]


@dataclass(frozen=True)
class _ImageSliceRequest:
    """A callable that stores all the input data needed to slice an image layer.

    This should be treated a deeply immutable structure, even though some
    fields can be modified in place. It is like a function that has captured
    all its inputs already.

    In general, the calling an instance of this may take a long time, so you may
    want to run it off the main thread.

    Attributes
    ----------
    dims : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    data : Any
        The layer's data field, which is the main input to slicing.
    indices : tuple of ints or slices
        The slice indices in the layer's data space.
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
    dask_indexer: DaskIndexer
    indices: Tuple[Union[int, slice], ...]
    multiscale: bool = field(repr=False)
    corner_pixels: np.ndarray
    rgb: bool = field(repr=False)
    data_level: int = field(repr=False)
    thumbnail_level: int = field(repr=False)
    level_shapes: np.ndarray = field(repr=False)
    downsample_factors: np.ndarray = field(repr=False)
    lazy: bool = field(default=False, repr=False)

    def __call__(self) -> _ImageSliceResponse:
        with self.dask_indexer():
            return (
                self._call_multi_scale()
                if self.multiscale
                else self._call_single_scale()
            )

    def _call_single_scale(self) -> _ImageSliceResponse:
        image = self.data[self.indices]
        if not self.lazy:
            image = np.asarray(image)
        # `Layer.multiscale` is mutable so we need to pass back the identity
        # transform to ensure `tile2data` is properly set on the layer.
        ndim = self.dims.ndim
        tile_to_data = Affine(
            name='tile2data', linear_matrix=np.eye(ndim), ndim=ndim
        )
        return _ImageSliceResponse(
            data=image,
            thumbnail=None,
            tile_to_data=tile_to_data,
            dims=self.dims,
            indices=self.indices,
        )

    def _call_multi_scale(self) -> _ImageSliceResponse:
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

        # Calculate the tile-to-data transform.
        scale = np.ones(self.dims.ndim)
        for d in self.dims.displayed:
            scale[d] = self.downsample_factors[level][d]

        translate = np.zeros(self.dims.ndim)
        if self.dims.ndisplay == 2:
            for d in self.dims.displayed:
                indices[d] = slice(
                    self.corner_pixels[0, d],
                    self.corner_pixels[1, d] + 1,
                    1,
                )
            translate = self.corner_pixels[0] * scale

        # This only needs to be a ScaleTranslate but different types
        # of transforms in a chain don't play nicely together right now.
        tile_to_data = Affine(
            name='tile2data',
            scale=scale,
            translate=translate,
            ndim=self.dims.ndim,
        )

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
            dims=self.dims,
            indices=self.indices,
        )

    def _slice_indices_at_level(
        self, level: int
    ) -> Tuple[Union[int, float, slice], ...]:
        indices = np.array(self.indices)
        axes = self.dims.not_displayed
        ds_indices = indices[axes] / self.downsample_factors[level][axes]
        ds_indices = np.round(ds_indices.astype(float)).astype(int)
        ds_indices = np.clip(ds_indices, 0, self.level_shapes[level][axes] - 1)
        indices[axes] = ds_indices
        return indices
