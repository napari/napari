from copy import copy, deepcopy
from typing import (
    Optional,
)

import numpy as np
import numpy.typing as npt
from scipy.stats import gmean

from napari.layers.base._base_constants import ActionType
from napari.layers.points._base import _BasePoints
from napari.layers.points._points_utils import (
    fix_data_points,
)
from napari.layers.points._slice import _PointSliceRequest
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice
from napari.layers.utils.layer_utils import (
    _features_to_properties,
)
from napari.utils.events.migrations import deprecation_warning_event
from napari.utils.migrations import add_deprecated_property, rename_argument
from napari.utils.transforms import Affine

DEFAULT_COLOR_CYCLE = np.array([[1, 0, 1, 1], [0, 1, 0, 1]])


class Points(_BasePoints):
    """Points layer.

    Parameters
    ----------
    data : array (N, D)
        Coordinates for N points in D dimensions.
    ndim : int
        Number of dimensions for shapes. When data is not None, ndim must be D.
        An empty points layer can be instantiated with arbitrary ndim.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    antialiasing: float
        Amount of antialiasing in canvas pixels.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', 'translucent_no_depth', 'additive', and 'minimum'}.
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
    border_width : float, array
        Width of the symbol border in pixels.
    border_width_is_relative : bool
        If enabled, border_width is interpreted as a fraction of the point size.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    canvas_size_limits : tuple of float
        Lower and upper limits for the size of points in canvas pixels.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.
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
    feature_defaults : dict[str, Any] or DataFrame
        The default value of each feature in a table with one row.
    features : dict[str, array-like] or DataFrame
        Features table where each row corresponds to a point and each column
        is a feature.
    metadata : dict
        Layer metadata.
    n_dimensional : bool
        This property will soon be deprecated in favor of 'out_of_slice_display'.
        Use that instead.
    name : str
        Name of the layer. If not provided then will be guessed using heuristics.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    out_of_slice_display : bool
        If True, renders points not just in central plane but also slightly out of slice
        according to specified point marker size.
    projection_mode : str
        How data outside the viewed dimensions but inside the thick Dims slice will
        be projected onto the viewed dimensions. Must fit to cls._projectionclass.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each point. Each property should be an array of length N,
        where N is the number of points.
    property_choices : dict {str: array (N,)}
        possible values for each property.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    scale : tuple of float
        Scale factors for the layer.
    shading : str, Shading
        Render lighting and shading on points. Options are:

        * 'none'
          No shading is added to the points.
        * 'spherical'
          Shading and depth buffer are changed to give a 3D spherical look to the points
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    shown : 1-D array of bool
        Whether to show each point.
    size : float, array
        Size of the point marker in data pixels. If given as a scalar, all points are made
        the same size. If given as an array, size must be the same or broadcastable
        to the same shape as the data.
    symbol : str, array
        Symbols to be used for the point markers. Must be one of the
        following: arrow, clobber, cross, diamond, disc, hbar, ring,
        square, star, tailed_arrow, triangle_down, triangle_up, vbar, x.
    text : str, dict
        Text to be displayed with the points. If text is set to a key in properties,
        the value of that property will be displayed. Multiple properties can be
        composed using f-string-like syntax (e.g., '{property_1}, {float_property:.2f}).
        A dictionary can be provided with keyword arguments to set the text values
        and display properties. See TextManager.__init__() for the valid keyword arguments.
        For example usage, see /napari/examples/add_points_with_text.py.
    translate : tuple of float
        Translation values for the layer.
    visible : bool
        Whether the layer visual is currently being displayed.

    Attributes
    ----------
    data : array (N, D)
        Coordinates for N points in D dimensions.
    features : DataFrame-like
        Features table where each row corresponds to a point and each column
        is a feature.
    feature_defaults : DataFrame-like
        Stores the default value of each feature in a table with one row.
    properties : dict {str: array (N,)} or DataFrame
        Annotations for each point. Each property should be an array of length N,
        where N is the number of points.
    text : str
        Text to be displayed with the points. If text is set to a key in properties, the value of
        that property will be displayed. Multiple properties can be composed using f-string-like
        syntax (e.g., '{property_1}, {float_property:.2f}).
        For example usage, see /napari/examples/add_points_with_text.py.
    symbol : array of str
        Array of symbols for each point.
    size : array (N,)
        Array of sizes for each point. Must have the same shape as the layer `data`.
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
        Border width of the marker for the next point to be added or the currently
        selected point.
    current_border_color : str
        Border color of the marker border for the next point to be added or the currently
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
        Border color setting mode.

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
        Whether each point is shown.

    Notes
    -----
    _view_data : array (M, D)
        coordinates of points in the currently viewed slice.
    _view_size : array (M, )
        Size of the point markers in the currently viewed slice.
    _view_symbol : array (M, )
        Symbols of the point markers in the currently viewed slice.
    _view_border_width : array (M, )
        Border width of the point markers in the currently viewed slice.
    _indices_view : array (M, )
        Integer indices of the points in the currently viewed slice and are shown.
    _selected_view :
        Integer indices of selected points in the currently viewed slice within
        the `_view_data` array.
    _selected_box : array (4, 2) or None
        Four corners of any box either around currently selected points or
        being created during a drag action. Starting in the top left and
        going clockwise.
    _drag_start : list or None
        Coordinates of first cursor click during a drag action. Gets reset to
        None after dragging is done.
    """

    @rename_argument(
        'edge_width', 'border_width', since_version='0.5.0', version='0.6.0'
    )
    @rename_argument(
        'edge_width_is_relative',
        'border_width_is_relative',
        since_version='0.5.0',
        version='0.6.0',
    )
    @rename_argument(
        'edge_color', 'border_color', since_version='0.5.0', version='0.6.0'
    )
    @rename_argument(
        'edge_color_cycle',
        'border_color_cycle',
        since_version='0.5.0',
        version='0.6.0',
    )
    @rename_argument(
        'edge_colormap',
        'border_colormap',
        since_version='0.5.0',
        version='0.6.0',
    )
    @rename_argument(
        'edge_contrast_limits',
        'border_contrast_limits',
        since_version='0.5.0',
        version='0.6.0',
    )
    def __init__(
        self,
        data=None,
        ndim=None,
        *,
        affine=None,
        antialiasing=1,
        blending='translucent',
        border_color='dimgray',
        border_color_cycle=None,
        border_colormap='viridis',
        border_contrast_limits=None,
        border_width=0.05,
        border_width_is_relative=True,
        cache=True,
        canvas_size_limits=(2, 10000),
        experimental_clipping_planes=None,
        face_color='white',
        face_color_cycle=None,
        face_colormap='viridis',
        face_contrast_limits=None,
        feature_defaults=None,
        features=None,
        metadata=None,
        n_dimensional=None,
        name=None,
        opacity=1.0,
        out_of_slice_display=False,
        projection_mode='none',
        properties=None,
        property_choices=None,
        rotate=None,
        scale=None,
        shading='none',
        shear=None,
        shown=True,
        size=10,
        symbol='o',
        text=None,
        translate=None,
        visible=True,
    ) -> None:
        if ndim is None:
            if scale is not None:
                ndim = len(scale)
            elif (
                data is not None
                and hasattr(data, 'shape')
                and len(data.shape) == 2
            ):
                ndim = data.shape[1]

        data, ndim = fix_data_points(data, ndim)

        # Save the point coordinates
        self._data = np.asarray(data)

        super().__init__(
            data,
            ndim=ndim,
            features=features,
            feature_defaults=feature_defaults,
            properties=properties,
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
            property_choices=property_choices,
            experimental_clipping_planes=experimental_clipping_planes,
            shading=shading,
            canvas_size_limits=canvas_size_limits,
            antialiasing=antialiasing,
            shown=shown,
            projection_mode=projection_mode,
        )

        deprecated_events = {}
        for attr in [
            '{}_width',
            'current_{}_width',
            '{}_width_is_relative',
            '{}_color',
            'current_{}_color',
        ]:
            old_attr = attr.format('edge')
            new_attr = attr.format('border')
            old_emitter = deprecation_warning_event(
                'layer.events',
                old_attr,
                new_attr,
                since_version='0.5.0',
                version='0.6.0',
            )
            getattr(self.events, new_attr).connect(old_emitter)
            deprecated_events[old_attr] = old_emitter

        self.events.add(**deprecated_events)

    @classmethod
    def _add_deprecated_properties(cls) -> None:
        """Adds deprecated properties to class."""
        deprecated_properties = [
            'edge_width',
            'edge_width_is_relative',
            'current_edge_width',
            'edge_color',
            'edge_color_cycle',
            'edge_colormap',
            'edge_contrast_limits',
            'current_edge_color',
            'edge_color_mode',
        ]
        for old_property in deprecated_properties:
            new_property = old_property.replace('edge', 'border')
            add_deprecated_property(
                cls,
                old_property,
                new_property,
                since_version='0.5.0',
                version='0.6.0',
            )

    @property
    def _points_data(self) -> np.ndarray:
        """Spatially distributed coordinates."""
        return self.data

    @property
    def data(self) -> np.ndarray:
        """(N, D) array: coordinates for N points in D dimensions."""
        return self._data

    @data.setter
    def data(self, data: Optional[np.ndarray]):
        """Set the data array and emit a corresponding event."""
        # Inheriting _BasePoints data.setter
        return _BasePoints.data.fset(self, data)

    def _set_data(self, data: Optional[np.ndarray]) -> None:
        """Set the .data array attribute, without emitting an event."""
        data, _ = fix_data_points(data, self.ndim)
        cur_npoints = len(self._data)
        self._data = data

        # Add/remove property and style values based on the number of new points.
        with (
            self.events.blocker_all(),
            self._border.events.blocker_all(),
            self._face.events.blocker_all(),
        ):
            self._feature_table.resize(len(data))
            self.text.apply(self.features)
            if len(data) < cur_npoints:
                # If there are now fewer p`oints, remove the size and colors of the
                # extra ones
                if len(self._border.colors) > len(data):
                    self._border._remove(
                        np.arange(len(data), len(self._border.colors))
                    )
                if len(self._face.colors) > len(data):
                    self._face._remove(
                        np.arange(len(data), len(self._face.colors))
                    )
                self._shown = self._shown[: len(data)]
                self._size = self._size[: len(data)]
                self._border_width = self._border_width[: len(data)]
                self._symbol = self._symbol[: len(data)]

            elif len(data) > cur_npoints:
                # If there are now more points, add the size and colors of the
                # new ones
                adding = len(data) - cur_npoints
                size = np.repeat(self.current_size, adding, axis=0)

                if len(self._border_width) > 0:
                    new_border_width = copy(self._border_width[-1])
                else:
                    new_border_width = self.current_border_width
                border_width = np.repeat([new_border_width], adding, axis=0)

                if len(self._symbol) > 0:
                    new_symbol = copy(self._symbol[-1])
                else:
                    new_symbol = self.current_symbol
                symbol = np.repeat([new_symbol], adding, axis=0)

                # Add new colors, updating the current property value before
                # to handle any in-place modification of feature_defaults.
                # Also see: https://github.com/napari/napari/issues/5634
                current_properties = self._feature_table.currents()
                self._border._update_current_properties(current_properties)
                self._border._add(n_colors=adding)
                self._face._update_current_properties(current_properties)
                self._face._add(n_colors=adding)

                shown = np.repeat([True], adding, axis=0)
                self._shown = np.concatenate((self._shown, shown), axis=0)

                self.size = np.concatenate((self._size, size), axis=0)
                self.border_width = np.concatenate(
                    (self._border_width, border_width), axis=0
                )
                self.symbol = np.concatenate((self._symbol, symbol), axis=0)

        self._update_dims()
        self._reset_editable()

    def _get_ndim(self) -> int:
        """Determine number of dimensions of the layer."""
        return self.data.shape[1]

    def _make_slice_request_internal(
        self, slice_input: _SliceInput, data_slice: _ThickNDSlice
    ) -> _PointSliceRequest:
        return _PointSliceRequest(
            slice_input=slice_input,
            data=self.data,
            data_slice=data_slice,
            projection_mode=self.projection_mode,
            out_of_slice_display=self.out_of_slice_display,
            size=self.size,
        )

    def add(self, coords):
        """Adds points at coordinates.

        Parameters
        ----------
        coords : array
            Point or points to add to the layer data.
        """
        cur_points = len(self.data)
        self.events.data(
            value=self.data,
            action=ActionType.ADDING,
            data_indices=(-1,),
            vertex_indices=((),),
        )
        self._set_data(np.append(self.data, np.atleast_2d(coords), axis=0))
        self.events.data(
            value=self.data,
            action=ActionType.ADDED,
            data_indices=(-1,),
            vertex_indices=((),),
        )
        self.selected_data = set(np.arange(cur_points, len(self.data)))

    def remove_selected(self) -> None:
        """Removes selected points if any."""
        index = list(self.selected_data)
        index.sort()
        if len(index):
            self.events.data(
                value=self.data,
                action=ActionType.REMOVING,
                data_indices=tuple(
                    self.selected_data,
                ),
                vertex_indices=((),),
            )
            self._shown = np.delete(self._shown, index, axis=0)
            self._size = np.delete(self._size, index, axis=0)
            self._symbol = np.delete(self._symbol, index, axis=0)
            self._border_width = np.delete(self._border_width, index, axis=0)
            with self._border.events.blocker_all():
                self._border._remove(indices_to_remove=index)
            with self._face.events.blocker_all():
                self._face._remove(indices_to_remove=index)
            self._feature_table.remove(index)
            self.text.remove(index)
            if self._value in self.selected_data:
                self._value = None
            else:
                if self._value is not None:
                    # update the index of self._value to account for the
                    # data being removed
                    indices_removed = np.array(index) < self._value
                    offset = np.sum(indices_removed)
                    self._value -= offset
                    self._value_stored -= offset

            self._set_data(np.delete(self.data, index, axis=0))
            self.events.data(
                value=self.data,
                action=ActionType.REMOVED,
                data_indices=tuple(
                    self.selected_data,
                ),
                vertex_indices=((),),
            )
            self.selected_data = set()

    def _move_points(
        self, ixgrid: tuple[np.ndarray, np.ndarray], shift: np.ndarray
    ) -> None:
        """Move points along a set a coordinates given a shift.

        Parameters
        ----------
        ixgrid : Tuple[np.ndarray, np.ndarray]
            Crossproduct indexing grid of node indices and dimensions, see `np.ix_`
        shift : np.ndarray
            Selected coordinates shift
        """
        self.data[ixgrid] = self.data[ixgrid] + shift

    def _paste_data(self) -> None:
        """Paste any point from clipboard and select them."""
        npoints = len(self._view_data)
        totpoints = len(self.data)

        if len(self._clipboard.keys()) > 0:
            not_disp = self._slice_input.not_displayed
            data = deepcopy(self._clipboard['data'])
            offset = [
                self._data_slice[i] - self._clipboard['indices'][i]
                for i in not_disp
            ]
            data[:, not_disp] = data[:, not_disp] + np.array(offset)
            self._data = np.append(self.data, data, axis=0)
            self._shown = np.append(
                self.shown, deepcopy(self._clipboard['shown']), axis=0
            )
            self._size = np.append(
                self.size, deepcopy(self._clipboard['size']), axis=0
            )
            self._symbol = np.append(
                self.symbol, deepcopy(self._clipboard['symbol']), axis=0
            )

            self._feature_table.append(self._clipboard['features'])

            self.text._paste(**self._clipboard['text'])

            self._border_width = np.append(
                self.border_width,
                deepcopy(self._clipboard['border_width']),
                axis=0,
            )
            self._border._paste(
                colors=self._clipboard['border_color'],
                properties=_features_to_properties(
                    self._clipboard['features']
                ),
            )
            self._face._paste(
                colors=self._clipboard['face_color'],
                properties=_features_to_properties(
                    self._clipboard['features']
                ),
            )

            self._selected_view = list(
                range(npoints, npoints + len(self._clipboard['data']))
            )
            self._selected_data.update(
                set(range(totpoints, totpoints + len(self._clipboard['data'])))
            )
            self.refresh()

    def _copy_data(self) -> None:
        """Copy selected points to clipboard."""
        if len(self.selected_data) > 0:
            index = list(self.selected_data)
            self._clipboard = {
                'data': deepcopy(self.data[index]),
                'border_color': deepcopy(self.border_color[index]),
                'face_color': deepcopy(self.face_color[index]),
                'shown': deepcopy(self.shown[index]),
                'size': deepcopy(self.size[index]),
                'symbol': deepcopy(self.symbol[index]),
                'border_width': deepcopy(self.border_width[index]),
                'features': deepcopy(self.features.iloc[index]),
                'indices': self._data_slice,
                'text': self.text._copy(index),
            }
        else:
            self._clipboard = {}

    def to_mask(
        self,
        *,
        shape: tuple,
        data_to_world: Optional[Affine] = None,
        isotropic_output: bool = True,
    ) -> npt.NDArray:
        """Return a binary mask array of all the points as balls.

        Parameters
        ----------
        shape : tuple
            The shape of the mask to be generated.
        data_to_world : Optional[Affine]
            The data-to-world transform of the output mask image. This likely comes from a reference image.
            If None, then this is the same as this layer's data-to-world transform.
        isotropic_output : bool
            If True, then force the output mask to always contain isotropic balls in data/pixel coordinates.
            Otherwise, allow the anisotropy in the data-to-world transform to squash the balls in certain dimensions.
            By default this is True, but you should set it to False if you are going to create a napari image
            layer from the result with the same data-to-world transform and want the visualized balls to be
            roughly isotropic.

        Returns
        -------
        np.ndarray
            The output binary mask array of the given shape containing this layer's points as balls.
        """
        if data_to_world is None:
            data_to_world = self._data_to_world
        mask = np.zeros(shape, dtype=bool)
        mask_world_to_data = data_to_world.inverse
        points_data_to_mask_data = self._data_to_world.compose(
            mask_world_to_data
        )
        points_in_mask_data_coords = np.atleast_2d(
            points_data_to_mask_data(self.data)
        )

        # Calculating the radii of the output points in the mask is complex.
        radii = self.size / 2

        # Scale each radius by the geometric mean scale of the Points layer to
        # keep the balls isotropic when visualized in world coordinates.
        # The geometric means are used instead of the arithmetic mean
        # to maintain the volume scaling factor of the transforms.
        point_data_to_world_scale = gmean(np.abs(self._data_to_world.scale))
        mask_world_to_data_scale = (
            gmean(np.abs(mask_world_to_data.scale))
            if isotropic_output
            else np.abs(mask_world_to_data.scale)
        )
        radii_scale = point_data_to_world_scale * mask_world_to_data_scale

        output_data_radii = radii[:, np.newaxis] * np.atleast_2d(radii_scale)

        for coords, radii in zip(
            points_in_mask_data_coords, output_data_radii
        ):
            # Define a minimal set of coordinates where the mask could be present
            # by defining an inclusive lower and exclusive upper bound for each dimension.
            lower_coords = np.maximum(np.floor(coords - radii), 0).astype(int)
            upper_coords = np.minimum(
                np.ceil(coords + radii) + 1, shape
            ).astype(int)
            # Generate every possible coordinate within the bounds defined above
            # in a grid of size D1 x D2 x ... x Dd x D (e.g. for D=2, this might be 4x5x2).
            submask_coords = [
                range(lower_coords[i], upper_coords[i])
                for i in range(self.ndim)
            ]
            submask_grids = np.stack(
                np.meshgrid(*submask_coords, copy=False, indexing='ij'),
                axis=-1,
            )
            # Update the mask coordinates based on the normalized square distance
            # using a logical or to maintain any existing positive mask locations.
            normalized_square_distances = np.sum(
                ((submask_grids - coords) / radii) ** 2, axis=-1
            )
            mask[np.ix_(*submask_coords)] |= normalized_square_distances <= 1
        return mask


Points._add_deprecated_properties()
