import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

from napari.utils.transforms import Affine
from napari.utils.translations import trans


@dataclass(frozen=True)
class _ImageSliceRequest:
    data: Any = field(repr=False)
    data_to_world: Affine = field(repr=False)
    ndim: int
    ndisplay: int
    point: Tuple[float, ...]
    dims_order: Tuple[int, ...]
    dims_displayed: Tuple[int, ...] = field(repr=False)
    dims_not_displayed: Tuple[int, ...] = field(repr=False)
    multiscale: bool = field(repr=False)
    corner_pixels: np.ndarray
    rgb: bool = field(repr=False)
    data_level: int = field(repr=False)
    thumbnail_level: int = field(repr=False)
    level_shapes: np.ndarray = field(repr=False)
    downsample_factors: np.ndarray = field(repr=False)
    lazy: bool = field(default=False, repr=False)


@dataclass(frozen=True)
class _ImageSliceResponse:
    data: Any = field(repr=False)
    thumbnail: Optional[Any] = field(repr=False)
    tile_to_data: Optional[Affine] = field(repr=False)


def _slice_image(
    request: _ImageSliceRequest, slice_indices
) -> _ImageSliceResponse:
    if request.multiscale:
        return _slice_image_multi_scale(request, slice_indices)
    return _slice_image_single_scale(request, slice_indices)


def _slice_image_single_scale(
    request: _ImageSliceRequest, slice_indices
) -> _ImageSliceResponse:
    image = request.data[slice_indices]
    if not request.lazy:
        image = np.asarray(image)
    return _ImageSliceResponse(
        data=image,
        thumbnail=None,
        tile_to_data=None,
    )


def _slice_image_multi_scale(
    request: _ImageSliceRequest, slice_indices
) -> _ImageSliceResponse:
    if request.ndisplay == 3:
        warnings.warn(
            trans._(
                'Multiscale rendering is only supported in 2D. In 3D, only the lowest resolution scale is displayed',
                deferred=True,
            ),
            category=UserWarning,
        )
        level = len(request.data) - 1
    else:
        level = request.data_level

    indices = _slice_indices_at_level(
        indices=slice_indices,
        level=level,
    )

    scale = np.ones(request.ndim)
    for d in request.dims_displayed:
        scale[d] = request.downsample_factors[level][d]

    # This only needs to be a ScaleTranslate but different types
    # of transforms in a chain don't play nicely together right now.
    tile_to_data = Affine(name='tile2data', scale=scale)
    if request.ndisplay == 2:
        for d in request.dims_displayed:
            indices[d] = slice(
                request.corner_pixels[0, d],
                request.corner_pixels[1, d],
                1,
            )
        # TODO: why do we only do this for 2D display?
        # I guess we only support multiscale in 2D anyway, but then why is scale
        # not in here too?
        tile_to_data.translate = request.corner_pixels[0] * tile_to_data.scale

    thumbnail_indices = _slice_indices_at_level(
        request=request,
        indices=slice_indices,
        level=request.thumbnail_level,
    )

    image = request.data[level][tuple(indices)]
    thumbnail = request.data[request.thumbnail_level][tuple(thumbnail_indices)]

    if not request.lazy:
        image = np.asarray(image)
        thumbnail = np.asarray(thumbnail)

    return _ImageSliceResponse(
        data=image,
        thumbnail=None,
        tile_to_data=None,
    )


def _slice_indices_at_level(
    *, request: _ImageSliceRequest, indices: Tuple, level: int
) -> np.ndarray:
    indices = np.array(indices)
    axes = request.not_displayed
    ds_indices = indices[axes] / request.downsample_factors[level][axes]
    ds_indices = np.round(ds_indices.astype(float)).astype(int)
    ds_indices = np.clip(ds_indices, 0, request.level_shapes[level][axes] - 1)
    indices[axes] = ds_indices
    return indices
