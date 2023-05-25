from dataclasses import dataclass, field
from typing import Any, Sequence, Tuple

import numpy as np

from napari.layers.utils._slice_input import _SliceInput

try:
    from napari_graph import BaseGraph
    from napari_graph.base_graph import _NODE_EMPTY_PTR

except ModuleNotFoundError:
    BaseGraph = None
    _NODE_EMPTY_PTR = None


@dataclass(frozen=True)
class _GraphSliceResponse:
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
    dims : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    """

    indices: np.ndarray = field(repr=False)
    edges_indices: np.ndarray = field(repr=False)
    scale: Any = field(repr=False)
    dims: _SliceInput


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
    dims_indices : tuple of ints or slices
        The slice indices in the layer's data space.
    size : array like
        Size of each node. This is used in calculating visibility.
    others
        See the corresponding attributes in `Layer` and `Image`.
    """

    dims: _SliceInput
    data: BaseGraph = field(repr=False)
    dims_indices: Any = field(repr=False)
    size: Any = field(repr=False)
    out_of_slice_display: bool = field(repr=False)

    def __call__(self) -> _GraphSliceResponse:
        # Return early if no data
        if self.data.n_nodes == 0:
            return _GraphSliceResponse(
                indices=[],
                edges_indices=[],
                scale=np.empty(0),
                dims=self.dims,
            )

        not_disp = list(self.dims.not_displayed)
        if not not_disp:
            # If we want to display everything, then use all indices.
            # scale is only impacted by not displayed data, therefore 1
            node_indices = np.arange(self.data.n_allocated_nodes)
            node_indices = node_indices[
                self.data._buffer2world != _NODE_EMPTY_PTR
            ]
            _, edges = self.data.get_edges_buffers(is_buffer_domain=True)
            return _GraphSliceResponse(
                indices=node_indices,
                edges_indices=edges,
                scale=1,
                dims=self.dims,
            )

        # We want a numpy array so we can use fancy indexing with the non-displayed
        # indices, but as self.dims_indices can (and often/always does) contain slice
        # objects, the array has dtype=object which is then very slow for the
        # arithmetic below. As Points._round_index is always False, we can safely
        # convert to float to get a major performance improvement.
        not_disp_indices = np.array(self.dims_indices)[not_disp].astype(float)

        if self.out_of_slice_display and self.dims.ndim > 2:
            (
                node_indices,
                edges_indices,
                scale,
            ) = self._get_out_of_display_slice_data(not_disp, not_disp_indices)
        else:
            node_indices, edges_indices, scale = self._get_slice_data(
                not_disp, not_disp_indices
            )

        return _GraphSliceResponse(
            indices=node_indices,
            edges_indices=edges_indices,
            scale=scale,
            dims=self.dims,
        )

    def _get_out_of_display_slice_data(
        self,
        not_disp: Sequence[int],
        not_disp_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Slices data according to non displayed indices
        and compute scaling factor for out-slice display
        while ignoring not initialized nodes from graph.
        """
        valid_nodes = self.data._buffer2world != _NODE_EMPTY_PTR
        ixgrid = np.ix_(valid_nodes, not_disp)
        data = self.data._coords[ixgrid]
        sizes = self.size[ixgrid] / 2
        distances = abs(data - not_disp_indices)
        matches = np.all(distances <= sizes, axis=1)
        size_match = sizes[matches]
        size_match[size_match == 0] = 1
        scale_per_dim = (size_match - distances[matches]) / size_match
        scale_per_dim[size_match == 0] = 1
        scale = np.prod(scale_per_dim, axis=1)
        valid_nodes[valid_nodes] = matches
        slice_indices = np.where(valid_nodes)[0].astype(int)
        edge_indices = self._valid_edges(valid_nodes)
        return slice_indices, edge_indices, scale

    def _get_slice_data(
        self,
        not_disp: Sequence[int],
        not_disp_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Slices data according to non displayed indices
        while ignoring not initialized nodes from graph.
        """
        valid_nodes = self.data._buffer2world != _NODE_EMPTY_PTR
        data = self.data._coords[np.ix_(valid_nodes, not_disp)]
        distances = np.abs(data - not_disp_indices)
        matches = np.all(distances <= 0.5, axis=1)
        valid_nodes[valid_nodes] = matches
        slice_indices = np.where(valid_nodes)[0].astype(int)
        edge_indices = self._valid_edges(valid_nodes)
        return slice_indices, edge_indices, 1

    def _valid_edges(
        self,
        nodes_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute edges (node pair) where both nodes are presents.

        Parameters
        ----------
        nodes_mask : np.ndarray
            Binary mask of available nodes.

        Returns
        -------
        np.ndarray
            (N x 2) array of nodes indices, where N is the number of valid edges.
        """
        _, edges = self.data.get_edges_buffers(is_buffer_domain=True)
        valid_edges = edges[
            np.logical_and(nodes_mask[edges[:, 0]], nodes_mask[edges[:, 1]])
        ]
        return valid_edges
