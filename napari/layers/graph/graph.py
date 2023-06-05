from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from napari.layers.graph._slice import _GraphSliceRequest, _GraphSliceResponse
from napari.layers.points.points import _BasePoints
from napari.layers.utils._slice_input import _SliceInput
from napari.utils.translations import trans

try:
    from napari_graph import BaseGraph, UndirectedGraph

except ModuleNotFoundError:
    BaseGraph = None
    UndirectedGraph = None


class Graph(_BasePoints):
    def __init__(
        self,
        data=None,
        *,
        ndim=None,
        features=None,
        feature_defaults=None,
        text=None,
        symbol='o',
        size=10,
        border_width=0.05,
        border_width_is_relative=True,
        border_color='dimgray',
        border_color_cycle=None,
        border_colormap='viridis',
        border_contrast_limits=None,
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
        experimental_clipping_planes=None,
        shading='none',
        canvas_size_limits=(2, 10000),
        antialiasing=1,
        shown=True,
    ) -> None:
        if BaseGraph is None:
            raise ImportError(
                trans._(
                    "`napari-graph` module is required by the graph layer."
                )
            )

        self._data = self._fix_data(data, ndim)
        self._edges_indices_view = []

        super().__init__(
            self._data,
            ndim=self._data.ndim,
            features=features,
            feature_defaults=feature_defaults,
            text=text,
            symbol=symbol,
            size=size,
            border_width=border_width,
            border_width_is_relative=border_width_is_relative,
            border_color=border_color,
            border_color_cycle=border_color_cycle,
            border_colormap=border_colormap,
            border_contrast_limits=border_contrast_limits,
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
            experimental_clipping_planes=experimental_clipping_planes,
            shading=shading,
            canvas_size_limits=canvas_size_limits,
            antialiasing=antialiasing,
            shown=shown,
        )

    @staticmethod
    def _fix_data(
        data: Optional[Union[BaseGraph, ArrayLike]] = None,
        ndim: Optional[int] = None,
    ) -> BaseGraph:
        """Checks input data and return a empty graph if is None."""
        if ndim is None:
            ndim = 2

        if data is None:
            # empty but pre-allocated graph
            return UndirectedGraph(ndim=ndim)

        if isinstance(data, BaseGraph):
            if data._coords is None:
                raise ValueError(
                    trans._(
                        "Graph layer must be a spatial graph, have the `coords` attribute."
                    )
                )
            return data

        try:
            arr_data = np.atleast_2d(data)
        except ValueError as err:
            raise NotImplementedError(
                trans._(
                    "Could not convert to {data} to a napari graph.",
                    data=data,
                )
            ) from err

        if not issubclass(arr_data.dtype.type, np.number):
            raise TypeError(
                trans._(
                    "Expected numeric type. Found{dtype}.",
                    dtype=arr_data.dtype,
                )
            )

        if arr_data.ndim > 2:
            raise ValueError(
                trans._(
                    "Graph layer only supports 2-dim arrays. Found {ndim}.",
                    ndim=arr_data.ndim,
                )
            )
        return UndirectedGraph(coords=arr_data)

    @property
    def _points_data(self) -> np.ndarray:
        return self._data._coords

    @property
    def data(self) -> BaseGraph:
        return self._data

    @data.setter
    def data(self, data: Union[BaseGraph, ArrayLike, None]) -> None:
        prev_size = self.data.n_allocated_nodes
        self._data = self._fix_data(data)
        self._data_changed(prev_size)

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

        indices = np.atleast_1d(indices)

        if len(coords) != len(indices):
            raise ValueError(
                trans._(
                    'coordinates and indices must have the same length. Found {coords_size} and {idx_size}',
                    coords_size=len(coords),
                    idx_size=len(indices),
                )
            )

        prev_size = self.data.n_allocated_nodes

        for idx, coord in zip(indices, coords):
            self.data.add_nodes(idx, coord)

        self._data_changed(prev_size)

    def remove_selected(self):
        """Removes selected points if any."""
        if len(self.selected_data):
            indices = self.data._buffer2world[list(self.selected_data)]
            self.remove(indices)
            self.selected_data = set()

    def remove(self, indices: ArrayLike) -> None:
        """Removes nodes given their indices."""
        indices = np.atleast_1d(indices)
        if indices.ndim > 1:
            raise ValueError(
                trans._(
                    "Indices for removal must be 1-dim. Found {ndim}",
                    ndim=indices.ndim,
                )
            )

        prev_size = self.data.n_allocated_nodes
        # descending order
        indices = np.flip(np.sort(indices))

        for idx in indices:
            self.data.remove_node(idx)

        self._data_changed(prev_size)

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

    def _update_props_and_style(self, data_size: int, prev_size: int) -> None:
        # Add/remove property and style values based on the number of new points.
        with self.events.blocker_all(), self._border.events.blocker_all(), self._face.events.blocker_all():
            self._feature_table.resize(data_size)
            self.text.apply(self.features)
            if data_size < prev_size:
                # If there are now fewer points, remove the size and colors of the
                # extra ones
                if len(self._border.colors) > data_size:
                    self._border._remove(
                        np.arange(data_size, len(self._border.colors))
                    )
                if len(self._face.colors) > data_size:
                    self._face._remove(
                        np.arange(data_size, len(self._face.colors))
                    )
                self._shown = self._shown[:data_size]
                self._size = self._size[:data_size]
                self._border_width = self._border_width[:data_size]
                self._symbol = self._symbol[:data_size]

            elif data_size > prev_size:
                adding = data_size - prev_size

                current_properties = self._feature_table.currents()
                self._border._update_current_properties(current_properties)
                self._border._add(n_colors=adding)
                self._face._update_current_properties(current_properties)
                self._face._add(n_colors=adding)

                for attribute in ("shown", "border_width", "symbol"):
                    if attribute == "shown":
                        default_value = True
                    else:
                        default_value = getattr(self, f"current_{attribute}")
                    new_values = np.repeat([default_value], adding, axis=0)
                    values = np.concatenate(
                        (getattr(self, f"_{attribute}"), new_values), axis=0
                    )
                    setattr(self, attribute, values)

                new_sizes = np.broadcast_to(
                    self.current_size, (adding, self._size.shape[1])
                )
                self.size = np.concatenate((self._size, new_sizes), axis=0)

    def _data_changed(self, prev_size: int) -> None:
        self._update_props_and_style(self.data.n_allocated_nodes, prev_size)
        self._update_dims()
        self.events.data(value=self.data)

    def _get_state(self) -> Dict[str, Any]:
        # FIXME: this method can be removed once 'properties' argument is deprecreated.
        state = super()._get_state()
        state.pop("properties", None)
        state.pop("property_choices", None)
        return state
