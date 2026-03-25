from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from napari.layers.base._slice import _next_request_id
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice


@dataclass(frozen=True)
class _ShapeSliceResponse:
    """Contains all the output data of slicing a Shapes layer.

    Attributes
    ----------
    indices : np.ndarray
        Indices of the shapes that are visible in the current slice.
    slice_key : np.ndarray
        The slice key (coordinates of non-displayed dimensions) used to
        compute visibility.
    slice_input : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    request_id : int
        The identifier of the request from which this was generated.
    """

    indices: np.ndarray = field(repr=False)
    slice_key: np.ndarray = field(repr=False)
    slice_input: _SliceInput
    request_id: int


@dataclass(frozen=True)
class _ShapeSliceRequest:
    """A callable that stores all the input data needed to slice a Shapes layer.

    This should be treated as a deeply immutable structure, even though some
    fields can be modified in place. It is like a function that has captured
    all its inputs already.

    In general, calling an instance of this may take a long time, so you may
    want to run it off the main thread.

    Attributes
    ----------
    slice_input : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    data_slice : _ThickNDSlice
        The slicing coordinates and margins in data space.
    slice_keys : Any
        The (N, 2, P) array of per-shape slice keys, where N is the number of
        shapes and P is the number of non-displayed dimensions. Each shape has
        a ``slice_key[0]`` (min values) and ``slice_key[1]`` (max values) for
        the non-displayed dimensions.
    id : int
        Unique identifier for this request.
    """

    slice_input: _SliceInput
    data_slice: _ThickNDSlice = field(repr=False)
    slice_keys: Any = field(repr=False)
    id: int = field(default_factory=_next_request_id)

    def __call__(self) -> _ShapeSliceResponse:
        not_disp = list(self.slice_input.not_displayed)
        n_shapes = len(self.slice_keys)

        if not not_disp:
            # All dims are displayed, all shapes are visible.
            return _ShapeSliceResponse(
                indices=np.arange(n_shapes, dtype=int),
                slice_key=np.array([]),
                slice_input=self.slice_input,
                request_id=self.id,
            )

        slice_key = np.array(self.data_slice.point)[not_disp]

        if n_shapes == 0:
            return _ShapeSliceResponse(
                indices=np.empty(0, dtype=int),
                slice_key=slice_key,
                slice_input=self.slice_input,
                request_id=self.id,
            )

        # If the shape slice_keys dimensions don't match the expected number of
        # non-displayed dimensions, it means ndisplay or dims_order changed and
        # the captured slice_keys are stale. In this case, we cannot compute
        # visibility here; _update_slice_response will recompute it after
        # updating the ShapeList's ndisplay/dims_order.
        expected_ndim = len(not_disp)
        if self.slice_keys.ndim < 3 or self.slice_keys.shape[2] != expected_ndim:
            return _ShapeSliceResponse(
                indices=np.empty(0, dtype=int),
                slice_key=slice_key,
                slice_input=self.slice_input,
                request_id=self.id,
            )

        # The slice key is repeated to check against both the min and max
        # values stored in each shape's slice key.
        # A shape is visible if its min <= slice_key <= max for all
        # non-displayed dimensions (with a tolerance of 0.5).
        slice_key_arr = np.array([slice_key, slice_key])
        displayed = np.all(
            np.abs(self.slice_keys - slice_key_arr) < 0.5, axis=(1, 2)
        )
        indices: npt.NDArray[np.intp] = np.where(displayed)[0].astype(int)

        return _ShapeSliceResponse(
            indices=indices,
            slice_key=slice_key,
            slice_input=self.slice_input,
            request_id=self.id,
        )
