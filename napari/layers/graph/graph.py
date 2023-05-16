from typing import Optional, Tuple

import numpy as np
from napari_graph import BaseGraph, UndirectedGraph
from numpy.typing import ArrayLike

from napari.layers.graph._slice import _GraphSliceRequest, _GraphSliceResponse
from napari.layers.points.points import _BasePoints
from napari.layers.utils._slice_input import _SliceInput
from napari.utils.translations import trans


class Graph(_BasePoints):
    def __init__(
        self,
        data=None,
        *,
        ndim=None,
        features=None,
        feature_defaults=None,
        properties=None,
        text=None,
        symbol='o',
        size=10,
        edge_width=0.05,
        edge_width_is_relative=True,
        edge_color='dimgray',
        edge_color_cycle=None,
        edge_colormap='viridis',
        edge_contrast_limits=None,
        face_color='white',
        face_color_cycle=None,
        face_colormap='viridis',
        face_contrast_limits=None,
        out_of_slice_display=False,
        n_dimensional=None,
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
        cache=True,
        property_choices=None,
        experimental_clipping_planes=None,
        shading='none',
        canvas_size_limits=(2, 10000),
        antialiasing=1,
        shown=True,
    ) -> None:
        self._data = self._fix_data(data, ndim)
        self._edges_indices_view = []

        super().__init__(
            data,
            ndim=self._data.ndim,
            features=features,
            feature_defaults=feature_defaults,
            properties=properties,
            text=text,
            symbol=symbol,
            size=size,
            edge_width=edge_width,
            edge_width_is_relative=edge_width_is_relative,
            edge_color=edge_color,
            edge_color_cycle=edge_color_cycle,
            edge_colormap=edge_colormap,
            edge_contrast_limits=edge_contrast_limits,
            face_color=face_color,
            face_color_cycle=face_color_cycle,
            face_colormap=face_colormap,
            face_contrast_limits=face_contrast_limits,
            out_of_slice_display=out_of_slice_display,
            n_dimensional=n_dimensional,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            rotate=rotate,
            shear=shear,
            affine=affine,
            opacity=opacity,
            blending=blending,
            visible=visible,
            cache=cache,
            property_choices=property_choices,
            experimental_clipping_planes=experimental_clipping_planes,
            shading=shading,
            canvas_size_limits=canvas_size_limits,
            antialiasing=antialiasing,
            shown=shown,
        )

    @staticmethod
    def _fix_data(
        data: Optional[BaseGraph] = None, ndim: int = 3
    ) -> BaseGraph:
        """Checks input data and return a empty graph if is None."""
        if data is None:
            return UndirectedGraph(n_nodes=100, ndim=ndim, n_edges=200)

        if isinstance(data, BaseGraph):
            return data

        raise NotImplementedError

    @property
    def _points_data(self) -> np.ndarray:
        return self._data._coords

    @property
    def data(self) -> BaseGraph:
        return self._data

    @data.setter
    def data(self, data: Optional[BaseGraph]) -> None:
        # FIXME: might be missing data changed call
        self._data = self._fix_data(data)

    def _get_ndim(self) -> int:
        """Determine number of dimensions of the layer."""
        return self.data.ndim

    def _make_slice_request_internal(
        self, slice_input: _SliceInput, dims_indices: ArrayLike
    ) -> _GraphSliceRequest:
        return _GraphSliceRequest(
            dims=slice_input,
            data=self.data,
            dims_indices=dims_indices,
            out_of_slice_display=self.out_of_slice_display,
            size=self.size,
        )

    def _update_slice_response(self, response: _GraphSliceResponse) -> None:
        super()._update_slice_response(response)
        self._edges_indices_view = response.edges_indices

    @property
    def _view_edges_coordinates(self) -> np.ndarray:
        return self.data._coords[self._edges_indices_view][
            ..., self._slice_input.displayed
        ]

    def add(
        self, coords: ArrayLike, indices: Optional[ArrayLike] = None
    ) -> None:
        """Adds nodes at coordinates.
        Parameters
        ----------
        coords : sequence of indices to add point at
        indices : optional indices of the newly inserted nodes.
        """
        coords = np.atleast_2d(coords)
        if indices is None:
            new_starting_idx = self.data._buffer2world.max() + 1
            indices = np.arange(
                new_starting_idx, new_starting_idx + len(coords)
            )

        if len(coords) != len(indices):
            raise ValueError(
                trans._(
                    'coordinates and indices must have the same length. Found {coords_size} and {idx_size}',
                    coords_size=len(coords),
                    idx_size=len(indices),
                )
            )

        for idx, coord in zip(indices, coords):
            self.data.add_nodes(idx, coord)

    def remove_selected(self):
        """Removes selected points if any."""
        if len(self.selected_data):
            indices = self.data._buffer2world[list(self.selected_data)]
            self.remove(indices)
            self.selected_data = set()

    def remove(self, indices: ArrayLike) -> None:
        """Removes nodes given their indices."""
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()

        indices.sort(reverse=True)
        for idx in indices:
            self.data.remove_node(idx)

    def _move_points(
        self, ixgrid: Tuple[np.ndarray, np.ndarray], shift: np.ndarray
    ) -> None:
        """Move points along a set a coordinates given a shift.

        Parameters
        ----------
        ixgrid : Tuple[np.ndarray, np.ndarray]
            Crossproduct indexing grid of node indices and dimensions, see `np.ix_`
        shift : np.ndarray
            Selected coordinates shift
        """
        self.data._coords[ixgrid] = self.data._coords[ixgrid] + shift
