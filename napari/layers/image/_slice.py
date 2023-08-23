from dataclasses import dataclass, field
from typing import Any, Tuple, Union

import numpy as np

from napari.layers.base._slice import _next_request_id
from napari.layers.utils._slice_input import _SliceInput
from napari.utils._dask_utils import DaskIndexer
from napari.utils.misc import reorder_after_dim_reduction
from napari.utils.transforms import Affine


@dataclass(frozen=True)
class _ImageSliceResponse:
    """Contains all the output data of slicing an image layer.

    Attributes
    ----------
    image : _ImageView
        The sliced image data.
    thumbnail: _ImageView
        The thumbnail image data. This may come from a different resolution to the sliced image
        data for multi-scale images. Otherwise, it's the same instance as data.
    tile_to_data: Affine
        The affine transform from the sliced data to the full data at the highest resolution.
        For single-scale images, this will be the identity matrix.
    dims : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    request_id : int
        The identifier of the request from which this was generated.
    """

    image: np.ndarray = field(repr=False)
    thumbnail: np.ndarray = field(repr=False)
    tile_to_data: Affine = field(repr=False)
    dims: _SliceInput
    request_id: int
    empty: bool = False

    @classmethod
    def make_empty(
        cls, *, dims: _SliceInput, rgb: bool
    ) -> '_ImageSliceResponse':
        """Returns an empty image slice response.

        An empty slice indicates that there is no valid slice data for an
        image layer, but allows other functionality that relies on slice
        data existing to continue to work without special casing.

        Parameters
        ----------
        dims : _SliceInput
            Describes the slicing plane or bounding box in the layer's dimensions.
        rgb : bool
            True if the underlying image is an RGB or RGBA image (i.e. that the
            last dimension represents a color channel that should not be sliced),
            False otherwise.
        """
        shape = (1,) * dims.ndisplay
        if rgb:
            shape = shape + (3,)
        data = np.zeros(shape)
        ndim = dims.ndim
        tile_to_data = Affine(
            name='tile2data', linear_matrix=np.eye(ndim), ndim=ndim
        )
        return _ImageSliceResponse(
            image=data,
            thumbnail=data,
            tile_to_data=tile_to_data,
            dims=dims,
            request_id=_next_request_id(),
            empty=True,
        )


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
    others
        See the corresponding attributes in `Layer` and `Image`.
    id : int
        The identifier of this slice request.
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
    id: int = field(default_factory=_next_request_id)

    def __call__(self) -> _ImageSliceResponse:
        with self.dask_indexer():
            return (
                self._call_multi_scale()
                if self.multiscale
                else self._call_single_scale()
            )

    def _call_single_scale(self) -> _ImageSliceResponse:
        order = self._get_order()
        data = np.asarray(self.data[self.indices])
        data = np.transpose(data, order)
        # `Layer.multiscale` is mutable so we need to pass back the identity
        # transform to ensure `tile2data` is properly set on the layer.
        ndim = self.dims.ndim
        tile_to_data = Affine(
            name='tile2data', linear_matrix=np.eye(ndim), ndim=ndim
        )
        return _ImageSliceResponse(
            image=data,
            thumbnail=data,
            tile_to_data=tile_to_data,
            dims=self.dims,
            request_id=self.id,
        )

    def _call_multi_scale(self) -> _ImageSliceResponse:
        if self.dims.ndisplay == 3:
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

        order = self._get_order()
        data = np.asarray(self.data[level][tuple(indices)])
        image = np.transpose(data, order)

        thumbnail = image
        if self.thumbnail_level != level:
            thumbnail_indices = self._slice_indices_at_level(
                self.thumbnail_level
            )
            thumbnail_data = np.asarray(
                self.data[self.thumbnail_level][tuple(thumbnail_indices)]
            )
            thumbnail = np.transpose(thumbnail_data, order)

        return _ImageSliceResponse(
            image=image,
            thumbnail=thumbnail,
            tile_to_data=tile_to_data,
            dims=self.dims,
            request_id=self.id,
        )

    def _slice_indices_at_level(self, level: int) -> np.ndarray:
        indices = np.array(self.indices)
        axes = self.dims.not_displayed
        ds_indices = indices[axes] / self.downsample_factors[level][axes]
        ds_indices = np.round(ds_indices.astype(float)).astype(int)
        ds_indices = np.clip(ds_indices, 0, self.level_shapes[level][axes] - 1)
        indices[axes] = ds_indices
        return indices

    def _get_order(self) -> Tuple[int, ...]:
        """Return the ordered displayed dimensions, but reduced to fit in the slice space."""
        order = reorder_after_dim_reduction(self.dims.displayed)
        if self.rgb:
            # if rgb need to keep the final axis fixed during the
            # transpose. The index of the final axis depends on how many
            # axes are displayed.
            return (*order, max(order) + 1)
        return order
