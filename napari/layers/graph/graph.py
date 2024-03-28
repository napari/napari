from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
from napari_graph import BaseGraph, UndirectedGraph, to_napari_graph
from numpy.typing import ArrayLike
from psygnal.containers import Selection

from napari.layers.base._base_constants import ActionType
from napari.layers.graph._slice import _GraphSliceRequest, _GraphSliceResponse
from napari.layers.points.points import _BasePoints
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice
from napari.utils.events import Event
from napari.utils.translations import trans


class Graph(_BasePoints):
    """
    Graph layer used to display spatial graphs.

    Parameters
    ----------
    data : GraphLike
        A napari-graph compatible data, for example, networkx graph, 2D array of
        coordinates or a napari-graph object.
    ndim : int
        Number of dimensions for shapes. When data is not None, ndim must be D.
        An empty points layer can be instantiated with arbitrary ndim.
    features : dict[str, array-like] or DataFrame
        Features table where each row corresponds to a point and each column
        is a feature.
    feature_defaults : dict[str, Any] or DataFrame
        The default value of each feature in a table with one row.
    text : str, dict
        Text to be displayed with the points. If text is set to a key in properties,
        the value of that property will be displayed. Multiple properties can be
        composed using f-string-like syntax (e.g., '{property_1}, {float_property:.2f}).
        A dictionary can be provided with keyword arguments to set the text values
        and display properties. See TextManager.__init__() for the valid keyword arguments.
        For example usage, see /napari/examples/add_points_with_text.py.
    symbol : str, array
        Symbols to be used for the point markers. Must be one of the
        following: arrow, clobber, cross, diamond, disc, hbar, ring,
        square, star, tailed_arrow, triangle_down, triangle_up, vbar, x.
    size : float, array
        Size of the point marker in data pixels. If given as a scalar, all points are made
        the same size. If given as an array, size must be the same or broadcastable
        to the same shape as the data.
    border_width : float, array
        Width of the symbol border in pixels.
    border_width_is_relative : bool
        If enabled, border_width is interpreted as a fraction of the point size.
    border_color : str, array-like, dict
        Color of the point marker border. Numeric color values should be RGB(A).
    border_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to border_color if a
        categorical attribute is used color the vectors.
    border_colormap : str, napari.utils.Colormap
        Colormap to set border_color if a continuous attribute is used to set face_color.
    border_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    face_color : str, array-like, dict
        Color of the point marker body. Numeric color values should be RGB(A).
    face_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to face_color if a
        categorical attribute is used color the vectors.
    face_colormap : str, napari.utils.Colormap
        Colormap to set face_color if a continuous attribute is used to set face_color.
    face_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    out_of_slice_display : bool
        If True, renders points not just in central plane but also slightly out of slice
        according to specified point marker size.
    n_dimensional : bool
        This property will soon be deprecated in favor of 'out_of_slice_display'.
        Use that instead.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    shading : str, Shading
        Render lighting and shading on points. Options are:

        * 'none'
          No shading is added to the points.
        * 'spherical'
          Shading and depth buffer are changed to give a 3D spherical look to the points
    antialiasing: float
        Amount of antialiasing in canvas pixels.
    canvas_size_limits : tuple of float
        Lower and upper limits for the size of points in canvas pixels.
    shown : 1-D array of bool
        Whether to show each point.

    Attributes
    ----------
    data : array (N, D)
        Coordinates for N points in D dimensions.
    features : DataFrame-like
        Features table where each row corresponds to a point and each column
        is a feature.
    feature_defaults : DataFrame-like
        Stores the default value of each feature in a table with one row.
    text : str
        Text to be displayed with the points. If text is set to a key in properties, the value of
        that property will be displayed. Multiple properties can be composed using f-string-like
        syntax (e.g., '{property_1}, {float_property:.2f}).
        For example usage, see /napari/examples/add_points_with_text.py.
    symbol : array of str
        Array of symbols for each point.
    size : array (N, D)
        Array of sizes for each point in each dimension. Must have the same
        shape as the layer `data`.
    border_width : array (N,)
        Width of the marker borders in pixels for all points
    border_width : array (N,)
        Width of the marker borders for all points as a fraction of their size.
    border_color : Nx4 numpy array
        Array of border color RGBA values, one for each point.
    border_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to border_color if a
        categorical attribute is used color the vectors.
    border_colormap : str, napari.utils.Colormap
        Colormap to set border_color if a continuous attribute is used to set face_color.
    border_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    face_color : Nx4 numpy array
        Array of face color RGBA values, one for each point.
    face_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to face_color if a
        categorical attribute is used color the vectors.
    face_colormap : str, napari.utils.Colormap
        Colormap to set face_color if a continuous attribute is used to set face_color.
    face_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    current_symbol : Symbol
        Symbol for the next point to be added or the currently selected points.
    current_size : float
        Size of the marker for the next point to be added or the currently
        selected point.
    current_border_width : float
        border width of the marker for the next point to be added or the currently
        selected point.
    current_border_color : str
        border color of the marker border for the next point to be added or the currently
        selected point.
    current_face_color : str
        Face color of the marker border for the next point to be added or the currently
        selected point.
    out_of_slice_display : bool
        If True, renders points not just in central plane but also slightly out of slice
        according to specified point marker size.
    selected_data : Selection
        Integer indices of any selected points.
    mode : str
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In ADD mode clicks of the cursor add points at the clicked location.

        In SELECT mode the cursor can select points by clicking on them or
        by dragging a box around them. Once selected points can be moved,
        have their properties edited, or be deleted.
    face_color_mode : str
        Face color setting mode.

        DIRECT (default mode) allows each point to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
    border_color_mode : str
        border color setting mode.

        DIRECT (default mode) allows each point to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
    shading : Shading
        Shading mode.
    antialiasing: float
        Amount of antialiasing in canvas pixels.
    canvas_size_limits : tuple of float
        Lower and upper limits for the size of points in canvas pixels.
    shown : 1-D array of bool
        Whether each node is shown.
    """

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
        projection_mode='none',
    ) -> None:
        self._data = self._fix_data(data, ndim)
        self._edges_indices_view: ArrayLike = []

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
            projection_mode=projection_mode,
        )

        # TODO:
        # dummy events because of VispyGraphLayer's VispyPointsLayerinheritance
        # should be removed in 0.6.0
        self.events.add(
            edge_width=Event,
            current_edge_width=Event,
            edge_width_is_relative=Event,
            edge_color=Event,
            current_edge_color=Event,
        )

    @staticmethod
    def _fix_data(
        data: Optional[Union[BaseGraph, ArrayLike]] = None,
        ndim: Optional[int] = None,
    ) -> BaseGraph:
        """Checks input data and return a empty graph if is None."""
        if data is None:
            if ndim is None:
                ndim = 2
            # empty but pre-allocated graph
            return UndirectedGraph(ndim=ndim)

        try:
            data = to_napari_graph(data)
        except NotImplementedError as e:
            raise TypeError from e

        if isinstance(data, BaseGraph):
            if not data.is_spatial():
                raise ValueError(
                    trans._(
                        'Graph layer must be a spatial graph, have the `coords` attribute (`pos` in NetworkX).'
                    )
                )
            return data

        return data

    @property
    def _points_data(self) -> np.ndarray:
        return self._data.coords_buffer

    @property
    def data(self) -> BaseGraph:
        return self._data

    @data.setter
    def data(self, data: np.ndarray) -> None:
        """Set the graphs data."""
        # Inhering _BasePoints data.setter
        return _BasePoints.data.fset(self, data)

    def _set_data(self, data: Union[BaseGraph, ArrayLike, None]) -> None:
        prev_size = self.data.n_allocated_nodes
        self._data = self._fix_data(data)
        self._data_changed(prev_size)

    def _get_ndim(self) -> int:
        """Determine number of dimensions of the layer."""
        return self.data.ndim

    def _make_slice_request_internal(
        self, slice_input: _SliceInput, data_slice: _ThickNDSlice
    ) -> _GraphSliceRequest:
        return _GraphSliceRequest(
            slice_input=slice_input,
            data=self.data,
            data_slice=data_slice,
            projection_mode=self.projection_mode,
            out_of_slice_display=self.out_of_slice_display,
            size=self.size,
        )

    def _update_slice_response(self, response: _GraphSliceResponse) -> None:  # type: ignore[override]
        super()._update_slice_response(response)
        self._edges_indices_view = response.edges_indices

    @property
    def _view_edges_coordinates(self) -> np.ndarray:
        return self.data.coords_buffer[self._edges_indices_view][
            ..., self._slice_input.displayed
        ]

    def add(
        self, coords: ArrayLike, indices: Optional[ArrayLike] = None
    ) -> None:
        """Adds nodes at coordinates.
        Parameters
        ----------
        coords : sequence of coordinates for each new node.
        indices : optional indices of the newly inserted nodes.
        """
        if indices is None:
            count_adding = len(np.atleast_2d(coords))
            indices = self.data.get_next_valid_indices(count_adding)
        indices = np.atleast_1d(indices)
        if indices.ndim > 1:
            raise ValueError(
                trans._(
                    'Indices for removal must be 1-dim. Found {ndim}',
                    ndim=indices.ndim,
                )
            )

        self.events.data(
            value=self.data,
            action=ActionType.ADDING,
            data_indices=tuple(indices),
            vertex_indices=((),),
        )

        prev_size = self.data.n_allocated_nodes
        added_indices = self.data.add_nodes(indices=indices, coords=coords)
        self._data_changed(prev_size)

        self.events.data(
            value=self.data,
            action=ActionType.ADDED,
            data_indices=tuple(
                added_indices,
            ),
            vertex_indices=((),),
        )
        self.selected_data = self.data._map_world2buffer(added_indices)

    def remove_selected(self) -> None:
        """Removes selected points if any."""
        if len(self.selected_data):
            self._remove_nodes(list(self.selected_data), is_buffer_domain=True)
            self.selected_data = cast(Selection[int], set())

    def remove(self, indices: ArrayLike) -> None:
        """Remove nodes given indices."""
        self._remove_nodes(indices, is_buffer_domain=False)

    def _remove_nodes(
        self,
        indices: ArrayLike,
        is_buffer_domain: bool,
    ) -> None:
        """
        Parameters
        ----------
        indices : ArrayLike
            List of node indices to remove.
        is_buffer_domain : bool
            Indicates if node indices are on world or buffer domain.
        """
        indices = np.atleast_1d(indices)
        if indices.ndim > 1:
            raise ValueError(
                trans._(
                    'Indices for removal must be 1-dim. Found {ndim}',
                    ndim=indices.ndim,
                )
            )
        # TODO: should know nothing about buffer
        world_indices = (
            self.data._buffer2world[indices] if is_buffer_domain else indices
        )
        self.events.data(
            value=self.data,
            action=ActionType.REMOVING,
            data_indices=tuple(
                world_indices,
            ),
            vertex_indices=((),),
        )

        prev_size = self.data.n_allocated_nodes

        # it got error missing __iter__ attribute, but we guarantee by np.atleast_1d call
        for idx in indices:  # type: ignore[union-attr]
            self.data.remove_node(idx, is_buffer_domain)

        self._data_changed(prev_size)

        self.events.data(
            value=self.data,
            action=ActionType.REMOVED,
            data_indices=tuple(
                world_indices,
            ),
            vertex_indices=((),),
        )

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
        self.data.coords_buffer[ixgrid] = (
            self.data.coords_buffer[ixgrid] + shift
        )

    def _update_props_and_style(self, data_size: int, prev_size: int) -> None:
        # Add/remove property and style values based on the number of new points.
        with (
            self.events.blocker_all(),
            self._border.events.blocker_all(),
            self._face.events.blocker_all(),
        ):
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

                # ensure each attribute is updated before refreshing
                with self._block_refresh():
                    for attribute in (
                        'shown',
                        'size',
                        'symbol',
                        'border_width',
                    ):
                        if attribute == 'shown':
                            default_value = True
                        else:
                            default_value = getattr(
                                self, f'current_{attribute}'
                            )
                        new_values = np.repeat([default_value], adding, axis=0)
                        values = np.concatenate(
                            (getattr(self, f'_{attribute}'), new_values),
                            axis=0,
                        )
                        setattr(self, attribute, values)

    def _data_changed(self, prev_size: int) -> None:
        self._update_props_and_style(self.data.n_allocated_nodes, prev_size)
        self._update_dims()

    def _get_state(self) -> Dict[str, Any]:
        # FIXME: this method can be removed once 'properties' argument is deprecreated.
        state = super()._get_state()
        state.pop('properties', None)
        state.pop('property_choices', None)
        return state
