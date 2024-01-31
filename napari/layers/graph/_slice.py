from dataclasses import dataclass, field
from typing import Any, Sequence, Tuple

import numpy as np
from napari_graph import BaseGraph
from numpy.typing import ArrayLike

from napari.layers.base._slice import _next_request_id
from napari.layers.points._points_constants import PointsProjectionMode
from napari.layers.points._slice import _PointSliceResponse
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice


@dataclass(frozen=True)
class _GraphSliceResponse(_PointSliceResponse):
    """Contains all the output data of slicing an graph layer.

    Attributes
    ----------
    indices : array like
        Indices of the sliced *nodes* data.
    edge_indices : array like
        Indices of the slice nodes for each *edge*.
    scale: array like or none
        Used to scale the sliced points for visualization.
        Should be broadcastable to indices.
    slice_input : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    request_id : int
        The identifier of the request from which this was generated.
    """

    edges_indices: ArrayLike = field(repr=False)


@dataclass(frozen=True)
class _GraphSliceRequest:
    """A callable that stores all the input data needed to slice a graph layer.

    This should be treated a deeply immutable structure, even though some
    fields can be modified in place. It is like a function that has captured
    all its inputs already.

    In general, the calling an instance of this may take a long time, so you may
    want to run it off the main thread.

    Attributes
    ----------
    dims : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    data : BaseGraph
        The layer's data field, which is the main input to slicing.
    data_slice : _ThickNDSlice
        The slicing coordinates and margins in data space.
    size : array like
        Size of each node. This is used in calculating visibility.
    others
        See the corresponding attributes in `Layer` and `Image`.
    """

    slice_input: _SliceInput
    data: BaseGraph = field(repr=False)
    data_slice: _ThickNDSlice = field(repr=False)
    projection_mode: PointsProjectionMode
    size: Any = field(repr=False)
    out_of_slice_display: bool = field(repr=False)
    id: int = field(default_factory=_next_request_id)

    def __call__(self) -> _GraphSliceResponse:
        # Return early if no data
        if self.data.n_nodes == 0:
            return _GraphSliceResponse(
                indices=[],
                edges_indices=[],
                scale=np.empty(0),
                slice_input=self.slice_input,
                request_id=self.id,
            )

        not_disp = list(self.slice_input.not_displayed)
        if not not_disp:
            # If we want to display everything, then use all indices.
            # scale is only impacted by not displayed data, therefore 1
            node_indices = np.arange(self.data.n_allocated_nodes)
            node_indices = node_indices[self.data.initialized_buffer_mask()]
            _, edges = self.data.get_edges_buffers(is_buffer_domain=True)
            return _GraphSliceResponse(
                indices=node_indices,
                edges_indices=edges,
                scale=1,
                slice_input=self.slice_input,
                request_id=self.id,
            )

        point, m_left, m_right = self.data_slice[not_disp].as_array()

        if self.projection_mode == 'none':
            low = point.copy()
            high = point.copy()
        else:
            low = point - m_left
            high = point + m_right

        # assume slice thickness of 1 in data pixels
        # (same as before thick slices were implemented)
        too_thin_slice = np.isclose(high, low)
        low[too_thin_slice] -= 0.5
        high[too_thin_slice] += 0.5

        in_slice, node_indices, edges_indices, scale = self._get_slice_data(
            not_disp, low, high
        )

        if self.out_of_slice_display and self.slice_input.ndim > 2:
            (
                node_indices,
                edges_indices,
                scale,
            ) = self._get_out_of_display_slice_data(
                not_disp, low, high, in_slice
            )

        return _GraphSliceResponse(
            indices=node_indices,
            edges_indices=edges_indices,
            scale=scale,
            slice_input=self.slice_input,
            request_id=self.id,
        )

    def _get_out_of_display_slice_data(
        self,
        not_disp: Sequence[int],
        low: np.ndarray,
        high: np.ndarray,
        in_slice: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, ArrayLike]:
        """
        Slices data according to non displayed indices
        and compute scaling factor for out-slice display
        while ignoring not initialized nodes from graph.
        """
        valid_nodes = self.data.initialized_buffer_mask()
        ixgrid = np.ix_(valid_nodes, not_disp)
        data = self.data.coords_buffer[ixgrid]
        sizes = self.size[valid_nodes, np.newaxis] / 2
        dist_from_low = np.abs(data - low)
        dist_from_high = np.abs(data - high)
        # keep distance of the closest margin
        distances = np.minimum(dist_from_low, dist_from_high)
        distances[in_slice] = 0
        matches = np.all(distances <= sizes, axis=1)
        if not np.any(matches):
            return np.empty(0, dtype=int), np.empty(0, dtype=int), 1
        size_match = sizes[matches]
        scale_per_dim = (size_match - distances[matches]) / size_match
        scale = np.prod(scale_per_dim, axis=1)
        valid_nodes[valid_nodes] = matches
        slice_indices = np.where(valid_nodes)[0].astype(int)
        edge_indices = self.data.subgraph_edges(
            slice_indices, is_buffer_domain=True
        )
        return slice_indices, edge_indices, scale

    def _get_slice_data(
        self,
        not_disp: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
    ) -> np.ndarray:
        """
        Slices data according to displayed indices
        while ignoring not initialized nodes from graph.
        """
        valid_nodes = self.data.initialized_buffer_mask()
        data = self.data.coords_buffer[np.ix_(valid_nodes, not_disp)]
        matches = np.all((data >= low) & (data <= high), axis=1)
        valid_nodes[valid_nodes] = matches
        slice_indices = np.where(valid_nodes)[0].astype(int)
        edge_indices = self.data.subgraph_edges(
            slice_indices, is_buffer_domain=True
        )
        return matches, slice_indices, edge_indices, 1
