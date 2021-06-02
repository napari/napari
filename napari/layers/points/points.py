import warnings
from copy import copy, deepcopy
from itertools import cycle
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

from ...utils.colormaps import Colormap, ValidColormapArg
from ...utils.colormaps.standardize_color import (
    get_color_namelist,
    hex_to_name,
    rgb_to_hex,
)
from ...utils.events import Event
from ...utils.translations import trans
from ..base import Layer
from ..utils._color_manager_constants import ColorMode
from ..utils.color_manager import ColorManager
from ..utils.color_transformations import ColorType
from ..utils.layer_utils import dataframe_to_properties
from ..utils.text import TextManager
from ._points_constants import SYMBOL_ALIAS, Mode, Symbol
from ._points_mouse_bindings import add, highlight, select
from ._points_utils import create_box, fix_data_points, points_to_squares

if TYPE_CHECKING:
    from pandas import DataFrame


DEFAULT_COLOR_CYCLE = np.array([[1, 0, 1, 1], [0, 1, 0, 1]])


class Points(Layer):
    """Points layer.

    Parameters
    ----------
    data : array (N, D)
        Coordinates for N points in D dimensions.
    ndim : int
        Number of dimensions for shapes. When data is not None, ndim must be D.
        An empty points layer can be instantiated with arbitrary ndim.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each point. Each property should be an array of length N,
        where N is the number of points.
    text : str, dict
        Text to be displayed with the points. If text is set to a key in properties,
        the value of that property will be displayed. Multiple properties can be
        composed using f-string-like syntax (e.g., '{property_1}, {float_property:.2f}).
        A dictionary can be provided with keyword arguments to set the text values
        and display properties. See TextManager.__init__() for the valid keyword arguments.
        For example usage, see /napari/examples/add_points_with_text.py.
    symbol : str
        Symbol to be used for the point markers. Must be one of the
        following: arrow, clobber, cross, diamond, disc, hbar, ring,
        square, star, tailed_arrow, triangle_down, triangle_up, vbar, x.
    size : float, array
        Size of the point marker. If given as a scalar, all points are made
        the same size. If given as an array, size must be the same
        broadcastable to the same shape as the data.
    edge_width : float
        Width of the symbol edge in pixels.
    edge_color : str, array-like, dict
        Color of the point marker border. Numeric color values should be RGB(A).
    edge_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to edge_color if a
        categorical attribute is used color the vectors.
    edge_colormap : str, napari.utils.Colormap
        Colormap to set edge_color if a continuous attribute is used to set face_color.
    edge_contrast_limits : None, (float, float)
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
    n_dimensional : bool
        If True, renders points not just in central plane but also in all
        n-dimensions according to specified point marker size.
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
        the final column is a lenght N translation vector and a 1 or a napari
        AffineTransform object. If provided then translate, scale, rotate, and
        shear values are ignored.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.
    property_choices : dict {str: array (N,)}
        possible values for each property.

    Attributes
    ----------
    data : array (N, D)
        Coordinates for N points in D dimensions.
    properties : dict {str: array (N,)} or DataFrame
        Annotations for each point. Each property should be an array of length N,
        where N is the number of points.
    text : str
        Text to be displayed with the points. If text is set to a key in properties, the value of
        that property will be displayed. Multiple properties can be composed using f-string-like
        syntax (e.g., '{property_1}, {float_property:.2f}).
        For example usage, see /napari/examples/add_points_with_text.py.
    symbol : str
        Symbol used for all point markers.
    size : array (N, D)
        Array of sizes for each point in each dimension. Must have the same
        shape as the layer `data`.
    edge_width : float
        Width of the marker edges in pixels for all points
    edge_color : Nx4 numpy array
        Array of edge color RGBA values, one for each point.
    edge_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to edge_color if a
        categorical attribute is used color the vectors.
    edge_colormap : str, napari.utils.Colormap
        Colormap to set edge_color if a continuous attribute is used to set face_color.
    edge_contrast_limits : None, (float, float)
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
    current_size : float
        Size of the marker for the next point to be added or the currently
        selected point.
    current_edge_color : str
        Size of the marker edge for the next point to be added or the currently
        selected point.
    current_face_color : str
        Size of the marker edge for the next point to be added or the currently
        selected point.
    n_dimensional : bool
        If True, renders points not just in central plane but also in all
        n-dimensions according to specified point marker size.
    selected_data : set
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
    edge_color_mode : str
        Edge color setting mode.

        DIRECT (default mode) allows each point to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute

    Notes
    -----
    _property_choices : dict {str: array (N,)}
        Possible values for the properties in Points.properties.
        If properties is not provided, it will be {} (empty dictionary).
    _view_data : array (M, 2)
        2D coordinates of points in the currently viewed slice.
    _view_size : array (M, )
        Size of the point markers in the currently viewed slice.
    _indices_view : array (M, )
        Integer indices of the points in the currently viewed slice.
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

    # TODO  write better documentation for edge_color and face_color

    # The max number of points that will ever be used to render the thumbnail
    # If more points are present then they are randomly subsampled
    _max_points_thumbnail = 1024

    def __init__(
        self,
        data=None,
        *,
        ndim=None,
        properties=None,
        text=None,
        symbol='o',
        size=10,
        edge_width=1,
        edge_color='black',
        edge_color_cycle=None,
        edge_colormap='viridis',
        edge_contrast_limits=None,
        face_color='white',
        face_color_cycle=None,
        face_colormap='viridis',
        face_contrast_limits=None,
        n_dimensional=False,
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
        property_choices=None,
    ):
        if ndim is None and scale is not None:
            ndim = len(scale)

        data, ndim = fix_data_points(data, ndim)

        super().__init__(
            data,
            ndim,
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
        )

        self.events.add(
            mode=Event,
            size=Event,
            edge_width=Event,
            face_color=Event,
            current_face_color=Event,
            edge_color=Event,
            current_edge_color=Event,
            properties=Event,
            current_properties=Event,
            symbol=Event,
            n_dimensional=Event,
            highlight=Event,
        )

        self._colors = get_color_namelist()

        # Save the point coordinates
        self._data = np.asarray(data)

        # Save the properties
        if self.data.size == 0 and properties:
            warnings.warn(
                trans._(
                    "Property choices should be passed as property_choices, not properties. This warning will become an error in version 0.4.11.",
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            property_choices = properties
            properties = {}
        self._properties, self._property_choices = self._prepare_properties(
            properties, property_choices, save_choices=True
        )

        # make the text
        if text is None or isinstance(text, (list, np.ndarray, str)):
            self._text = TextManager(text, len(data), self.properties)
        elif isinstance(text, dict):
            copied_text = deepcopy(text)
            copied_text['properties'] = self.properties
            copied_text['n_text'] = len(data)
            self._text = TextManager(**copied_text)
        else:
            raise TypeError(
                trans._(
                    'text should be a string, array, or dict',
                    deferred=True,
                )
            )

        # Save the point style params
        self.symbol = symbol
        self._n_dimensional = n_dimensional
        self.edge_width = edge_width

        # The following point properties are for the new points that will
        # be added. For any given property, if a list is passed to the
        # constructor so each point gets its own value then the default
        # value is used when adding new points
        self._current_size = np.asarray(size) if np.isscalar(size) else 10
        # Indices of selected points
        self._selected_data = set()
        self._selected_data_stored = set()
        self._selected_data_history = set()
        # Indices of selected points within the currently viewed slice
        self._selected_view = []
        # Index of hovered point
        self._value = None
        self._value_stored = None
        self._mode = Mode.PAN_ZOOM
        self._mode_history = self._mode
        self._status = self.mode
        self._highlight_index = []
        self._highlight_box = None

        self._drag_start = None

        # initialize view data
        self._indices_view = np.empty(0)
        self._view_size_scale = []

        self._drag_box = None
        self._drag_box_stored = None
        self._is_selecting = False
        self._clipboard = {}
        self._round_index = False

        self._edge = ColorManager._from_layer_kwargs(
            n_colors=len(data),
            colors=edge_color,
            continuous_colormap=edge_colormap,
            contrast_limits=edge_contrast_limits,
            categorical_colormap=edge_color_cycle,
            properties=self._properties
            if self._data.size
            else self._property_choices,
        )
        self._face = ColorManager._from_layer_kwargs(
            n_colors=len(data),
            colors=face_color,
            continuous_colormap=face_colormap,
            contrast_limits=face_contrast_limits,
            categorical_colormap=face_color_cycle,
            properties=self._properties
            if self._data.size
            else self._property_choices,
        )

        self.size = size

        # set the current_properties
        if len(data) > 0:
            self.current_properties = {
                k: np.asarray([v[-1]]) for k, v in self.properties.items()
            }
        elif len(data) == 0 and self.properties:
            self.current_properties = {
                k: np.asarray([v[0]])
                for k, v in self._property_choices.items()
            }
        else:
            self.current_properties = {}

        # Trigger generation of view slice and thumbnail
        self._update_dims()

    @property
    def data(self) -> np.ndarray:
        """(N, D) array: coordinates for N points in D dimensions."""
        return self._data

    @data.setter
    def data(self, data: Optional[np.ndarray]):
        data, _ = fix_data_points(data, self.ndim)
        cur_npoints = len(self._data)
        self._data = data

        # Adjust the size array when the number of points has changed
        with self.events.blocker_all():
            with self._edge.events.blocker_all():
                with self._face.events.blocker_all():
                    if len(data) < cur_npoints:
                        # If there are now fewer points, remove the size and colors of the
                        # extra ones
                        if len(self._edge.colors) > len(data):
                            self._edge._remove(
                                np.arange(len(data), len(self._edge.colors))
                            )
                        if len(self._face.colors) > len(data):
                            self._face._remove(
                                np.arange(len(data), len(self._face.colors))
                            )
                        self._size = self._size[: len(data)]

                        for k in self.properties:
                            self.properties[k] = self.properties[k][
                                : len(data)
                            ]

                    elif len(data) > cur_npoints:
                        # If there are now more points, add the size and colors of the
                        # new ones
                        adding = len(data) - cur_npoints
                        if len(self._size) > 0:
                            new_size = copy(self._size[-1])
                            for i in self._dims_displayed:
                                new_size[i] = self.current_size
                        else:
                            # Add the default size, with a value for each dimension
                            new_size = np.repeat(
                                self.current_size, self._size.shape[1]
                            )
                        size = np.repeat([new_size], adding, axis=0)

                        for k in self.properties:
                            new_property = np.repeat(
                                self.current_properties[k], adding, axis=0
                            )
                            self.properties[k] = np.concatenate(
                                (self.properties[k], new_property), axis=0
                            )

                        # add new colors
                        self._edge._add(n_colors=adding)
                        self._face._add(n_colors=adding)

                        self.size = np.concatenate((self._size, size), axis=0)
                        self.selected_data = set(
                            np.arange(cur_npoints, len(data))
                        )

                        self.text.add(self.current_properties, adding)

        self._update_dims()
        self.events.data(value=self.data)
        self._set_editable()

    def _on_selection(self, selected):
        if selected:
            self._set_highlight()
        else:
            self._highlight_box = None
            self._highlight_index = []
            self.events.highlight()

    @property
    def property_choices(self) -> Dict[str, np.ndarray]:
        return self._property_choices

    @property
    def properties(self) -> Dict[str, np.ndarray]:
        """dict {str: np.ndarray (N,)}, DataFrame: Annotations for each point"""
        return self._properties

    @staticmethod
    def _update_color_manager(
        color_manager, properties, current_properties, name
    ):
        if color_manager.color_properties is not None:
            if color_manager.color_properties.name not in properties:
                color_manager.color_mode = ColorMode.DIRECT
                color_manager.color_properties = None
                warnings.warn(
                    trans._(
                        'property used for {name} dropped',
                        deferred=True,
                        name=name,
                    ),
                    RuntimeWarning,
                )
            else:
                color_name = color_manager.color_properties.name
                color_manager.color_properties = {
                    'name': color_name,
                    'values': properties[color_name],
                    'current_value': current_properties[color_name],
                }

    @properties.setter
    def properties(
        self, properties: Union[Dict[str, np.ndarray], 'DataFrame', None]
    ):
        self._properties, self._property_choices = self._prepare_properties(
            properties, self._property_choices
        )
        self._update_color_manager(
            self._face,
            self._properties,
            self._current_properties,
            "face_color",
        )
        self._update_color_manager(
            self._edge,
            self._properties,
            self._current_properties,
            "edge_color",
        )

        if self.text.values is not None:
            self.refresh_text()
        self.events.properties()

    def _prepare_properties(
        self,
        properties: Union[
            Dict[str, Union[np.ndarray, list]], 'DataFrame', None
        ],
        property_choices: Dict[str, Union[np.ndarray, list]] = None,
        save_choices: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Return properties in a normalized dict-of-columns format.

        Parameters
        ----------
        properties : Union[dict, DataFrame]
            properties to be transformed
        property_choices : Dict[str, np.ndarray]
            previous choices
        save_choices : bool
            preserve property choices that are not available in input columns.

        Returns
        -------
        properties (dict):
            properties dictionary
        """
        if property_choices is None:
            property_choices = {}
        if properties is None:
            properties = {}
        if not isinstance(properties, dict):
            properties, _ = dataframe_to_properties(properties)

        new_choices = {
            k: np.unique(np.concatenate((v, property_choices.get(k, []))))
            for k, v in properties.items()
        }
        if not new_choices:
            # case of set empty properties when have available choices list
            new_choices = {
                k: np.unique(v) for k, v in property_choices.items()
            }
        if not properties and new_choices:
            if self._data.size:
                properties = {
                    k: [None] * self._data.shape[0] for k in new_choices
                }
            else:
                properties = {
                    k: np.empty(0, v.dtype) for k, v in new_choices.items()
                }
        if save_choices:
            for k, v in property_choices.items():
                if k not in new_choices:
                    new_choices[k] = np.unique(v)
                    properties[k] = [None] * self._data.shape[0]
        return self._validate_properties(properties), new_choices

    @property
    def current_properties(self) -> Dict[str, np.ndarray]:
        """dict{str: np.ndarray(1,)}: properties for the next added point."""
        return self._current_properties

    @current_properties.setter
    def current_properties(self, current_properties):
        self._current_properties = current_properties

        if (
            self._update_properties
            and len(self.selected_data) > 0
            and self._mode != Mode.ADD
        ):
            props = self.properties
            for k in props:
                props[k][list(self.selected_data)] = current_properties[k]
            self.properties = props

        self._edge._update_current_properties(current_properties)
        self._face._update_current_properties(current_properties)
        self.events.current_properties()

    def _validate_properties(
        self, properties: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Validates the type and size of the properties"""
        for k, v in properties.items():
            if len(v) != len(self.data):
                raise ValueError(
                    trans._(
                        'the number of properties must equal the number of points',
                        deferred=True,
                    )
                )
            # ensure the property values are a numpy array
            if type(v) != np.ndarray:
                properties[k] = np.asarray(v)

        return properties

    @property
    def text(self) -> TextManager:
        """TextManager: the TextManager object containing containing the text properties"""
        return self._text

    @text.setter
    def text(self, text):
        self._text._set_text(
            text, n_text=len(self.data), properties=self.properties
        )

    def refresh_text(self):
        """Refresh the text values.

        This is generally used if the properties were updated without changing the data
        """
        self.text.refresh_text(self.properties)

    def _get_ndim(self) -> int:
        """Determine number of dimensions of the layer."""
        return self.data.shape[1]

    @property
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        if len(self.data) == 0:
            extrema = np.full((2, self.ndim), np.nan)
        else:
            maxs = np.max(self.data, axis=0)
            mins = np.min(self.data, axis=0)
            extrema = np.vstack([mins, maxs])
        return extrema

    @property
    def n_dimensional(self) -> bool:
        """bool: renders points as n-dimensionsal."""
        return self._n_dimensional

    @n_dimensional.setter
    def n_dimensional(self, n_dimensional: bool) -> None:
        self._n_dimensional = n_dimensional
        self.events.n_dimensional()
        self.refresh()

    @property
    def symbol(self) -> str:
        """str: symbol used for all point markers."""
        return str(self._symbol)

    @symbol.setter
    def symbol(self, symbol: Union[str, Symbol]) -> None:

        if isinstance(symbol, str):
            # Convert the alias string to the deduplicated string
            if symbol in SYMBOL_ALIAS:
                symbol = SYMBOL_ALIAS[symbol]
            else:
                symbol = Symbol(symbol)
        self._symbol = symbol
        self.events.symbol()
        self.events.highlight()

    @property
    def size(self) -> Union[int, float, np.ndarray, list]:
        """(N, D) array: size of all N points in D dimensions."""
        return self._size

    @size.setter
    def size(self, size: Union[int, float, np.ndarray, list]) -> None:
        try:
            self._size = np.broadcast_to(size, self.data.shape).copy()
        except Exception:
            try:
                self._size = np.broadcast_to(
                    size, self.data.shape[::-1]
                ).T.copy()
            except Exception:
                raise ValueError(
                    trans._(
                        "Size is not compatible for broadcasting",
                        deferred=True,
                    )
                )
        self.refresh()

    @property
    def current_size(self) -> Union[int, float]:
        """float: size of marker for the next added point."""
        return self._current_size

    @current_size.setter
    def current_size(self, size: Union[None, float]) -> None:
        self._current_size = size
        if (
            self._update_properties
            and len(self.selected_data) > 0
            and self._mode != Mode.ADD
        ):
            for i in self.selected_data:
                self.size[i, :] = (self.size[i, :] > 0) * size
            self.refresh()
            self.events.size()

    @property
    def edge_width(self) -> Union[None, int, float]:
        """float: width used for all point markers."""
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: Union[None, float]) -> None:
        self._edge_width = edge_width
        self.events.edge_width()

    @property
    def edge_color(self) -> np.ndarray:
        """(N x 4) np.ndarray: Array of RGBA edge colors for each point"""
        return self._edge.colors

    @edge_color.setter
    def edge_color(self, edge_color):
        self._edge._set_color(
            color=edge_color,
            n_colors=len(self.data),
            properties=self.properties,
            current_properties=self.current_properties,
        )
        self.events.edge_color()

    @property
    def edge_color_cycle(self) -> np.ndarray:
        """Union[list, np.ndarray] :  Color cycle for edge_color.
        Can be a list of colors defined by name, RGB or RGBA
        """
        return self._edge.categorical_colormap.fallback_color.values

    @edge_color_cycle.setter
    def edge_color_cycle(self, edge_color_cycle: Union[list, np.ndarray]):
        self._edge.categorical_colormap = edge_color_cycle

    @property
    def edge_colormap(self) -> Colormap:
        """Return the colormap to be applied to a property to get the edge color.

        Returns
        -------
        colormap : napari.utils.Colormap
            The Colormap object.
        """
        return self._edge.continuous_colormap

    @edge_colormap.setter
    def edge_colormap(self, colormap: ValidColormapArg):
        self._edge.continuous_colormap = colormap

    @property
    def edge_contrast_limits(self) -> Tuple[float, float]:
        """None, (float, float): contrast limits for mapping
        the edge_color colormap property to 0 and 1
        """
        return self._edge.contrast_limits

    @edge_contrast_limits.setter
    def edge_contrast_limits(
        self, contrast_limits: Union[None, Tuple[float, float]]
    ):
        self._edge.contrast_limits = contrast_limits

    @property
    def current_edge_color(self) -> str:
        """str: Edge color of marker for the next added point or the selected point(s)."""
        hex_ = rgb_to_hex(self._edge.current_color)[0]
        return hex_to_name.get(hex_, hex_)

    @current_edge_color.setter
    def current_edge_color(self, edge_color: ColorType) -> None:
        if (
            self._update_properties
            and len(self.selected_data) > 0
            and self._mode != Mode.ADD
        ):
            update_indices = list(self.selected_data)
        else:
            update_indices = []
        self._edge._update_current_color(
            edge_color, update_indices=update_indices
        )
        self.events.current_edge_color()

    @property
    def edge_color_mode(self) -> str:
        """str: Edge color setting mode

        DIRECT (default mode) allows each point to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
        """
        return self._edge.color_mode

    @edge_color_mode.setter
    def edge_color_mode(self, edge_color_mode: Union[str, ColorMode]):
        self._set_color_mode(edge_color_mode, 'edge')

    @property
    def face_color(self) -> np.ndarray:
        """(N x 4) np.ndarray: Array of RGBA face colors for each point"""
        return self._face.colors

    @face_color.setter
    def face_color(self, face_color):
        self._face._set_color(
            color=face_color,
            n_colors=len(self.data),
            properties=self.properties,
            current_properties=self.current_properties,
        )
        self.events.face_color()

    @property
    def face_color_cycle(self) -> np.ndarray:
        """Union[np.ndarray, cycle]:  Color cycle for face_color
        Can be a list of colors defined by name, RGB or RGBA
        """
        return self._face.categorical_colormap.fallback_color.values

    @face_color_cycle.setter
    def face_color_cycle(self, face_color_cycle: Union[np.ndarray, cycle]):
        self._face.categorical_colormap = face_color_cycle

    @property
    def face_colormap(self) -> Colormap:
        """Return the colormap to be applied to a property to get the face color.

        Returns
        -------
        colormap : napari.utils.Colormap
            The Colormap object.
        """
        return self._face.continuous_colormap

    @face_colormap.setter
    def face_colormap(self, colormap: ValidColormapArg):
        self._face.continuous_colormap = colormap

    @property
    def face_contrast_limits(self) -> Union[None, Tuple[float, float]]:
        """None, (float, float) : clims for mapping the face_color
        colormap property to 0 and 1
        """
        return self._face.contrast_limits

    @face_contrast_limits.setter
    def face_contrast_limits(
        self, contrast_limits: Union[None, Tuple[float, float]]
    ):
        self._face.contrast_limits = contrast_limits

    @property
    def current_face_color(self) -> str:
        """Face color of marker for the next added point or the selected point(s)."""
        hex_ = rgb_to_hex(self._face.current_color)[0]
        return hex_to_name.get(hex_, hex_)

    @current_face_color.setter
    def current_face_color(self, face_color: ColorType) -> None:

        if (
            self._update_properties
            and len(self.selected_data) > 0
            and self._mode != Mode.ADD
        ):
            update_indices = list(self.selected_data)
        else:
            update_indices = []
        self._face._update_current_color(
            face_color, update_indices=update_indices
        )
        self.events.current_face_color()

    @property
    def face_color_mode(self) -> str:
        """str: Face color setting mode

        DIRECT (default mode) allows each point to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
        """
        return self._face.color_mode

    @face_color_mode.setter
    def face_color_mode(self, face_color_mode):
        self._set_color_mode(face_color_mode, 'face')

    def _set_color_mode(
        self, color_mode: Union[ColorMode, str], attribute: str
    ):
        """Set the face_color_mode or edge_color_mode property

        Parameters
        ----------
        color_mode : str, ColorMode
            The value for setting edge or face_color_mode. If color_mode is a string,
            it should be one of: 'direct', 'cycle', or 'colormap'
        attribute : str in {'edge', 'face'}
            The name of the attribute to set the color of.
            Should be 'edge' for edge_colo_moder or 'face' for face_color_mode.
        """
        color_mode = ColorMode(color_mode)
        color_manager = getattr(self, f'_{attribute}')

        if color_mode == ColorMode.DIRECT:
            color_manager.color_mode = color_mode
        elif color_mode in (ColorMode.CYCLE, ColorMode.COLORMAP):
            if color_manager.color_properties is not None:
                color_property = color_manager.color_properties.name
            else:
                color_property = ''
            if color_property == '':
                if self.properties:
                    new_color_property = next(iter(self.properties))
                    color_manager.color_properties = {
                        'name': new_color_property,
                        'values': self.properties[new_color_property],
                        'current_value': np.squeeze(
                            self.current_properties[new_color_property]
                        ),
                    }
                    warnings.warn(
                        trans._(
                            '_{attribute}_color_property was not set, setting to: {new_color_property}',
                            deferred=True,
                            attribute=attribute,
                            new_color_property=new_color_property,
                        )
                    )
                else:
                    raise ValueError(
                        trans._(
                            'There must be a valid Points.properties to use {color_mode}',
                            deferred=True,
                            color_mode=color_mode,
                        )
                    )

            # ColorMode.COLORMAP can only be applied to numeric properties
            color_property = color_manager.color_properties.name
            if (color_mode == ColorMode.COLORMAP) and not issubclass(
                self.properties[color_property].dtype.type, np.number
            ):
                raise TypeError(
                    trans._(
                        'selected property must be numeric to use ColorMode.COLORMAP',
                        deferred=True,
                    )
                )
            color_manager.color_mode = color_mode

    def refresh_colors(self, update_color_mapping: bool = False):
        """Calculate and update face and edge colors if using a cycle or color map
        Parameters
        ----------
        update_color_mapping : bool
            If set to True, the function will recalculate the color cycle map
            or colormap (whichever is being used). If set to False, the function
            will use the current color cycle map or color map. For example, if you
            are adding/modifying points and want them to be colored with the same
            mapping as the other points (i.e., the new points shouldn't affect
            the color cycle map or colormap), set update_color_mapping=False.
            Default value is False.
        """
        self._edge._refresh_colors(self.properties, update_color_mapping)
        self._face._refresh_colors(self.properties, update_color_mapping)

    def _get_state(self):
        """Get dictionary of layer state.

        Returns
        -------
        state : dict
            Dictionary of layer state.
        """
        state = self._get_base_state()
        state.update(
            {
                'symbol': self.symbol,
                'edge_width': self.edge_width,
                'face_color': self.face_color,
                'face_color_cycle': self.face_color_cycle,
                'face_colormap': self.face_colormap.name,
                'face_contrast_limits': self.face_contrast_limits,
                'edge_color': self.edge_color,
                'edge_color_cycle': self.edge_color_cycle,
                'edge_colormap': self.edge_colormap.name,
                'edge_contrast_limits': self.edge_contrast_limits,
                'properties': self.properties,
                'property_choices': self._property_choices,
                'text': self.text._get_state(),
                'n_dimensional': self.n_dimensional,
                'size': self.size,
                'ndim': self.ndim,
                'data': self.data,
            }
        )
        return state

    @property
    def selected_data(self) -> set:
        """set: set of currently selected points."""
        return self._selected_data

    @selected_data.setter
    def selected_data(self, selected_data):
        self._selected_data = set(selected_data)
        self._selected_view = list(
            np.intersect1d(
                np.array(list(self._selected_data)),
                self._indices_view,
                return_indices=True,
            )[2]
        )

        # Update properties based on selected points
        if not len(self._selected_data):
            self._set_highlight()
            return
        index = list(self._selected_data)
        edge_colors = np.unique(self.edge_color[index], axis=0)
        if len(edge_colors) == 1:
            edge_color = edge_colors[0]
            with self.block_update_properties():
                self.current_edge_color = edge_color

        face_colors = np.unique(self.face_color[index], axis=0)
        if len(face_colors) == 1:
            face_color = face_colors[0]
            with self.block_update_properties():
                self.current_face_color = face_color

        size = list({self.size[i, self._dims_displayed].mean() for i in index})
        if len(size) == 1:
            size = size[0]
            with self.block_update_properties():
                self.current_size = size

        properties = {
            k: np.unique(v[index], axis=0) for k, v in self.properties.items()
        }
        n_unique_properties = np.array([len(v) for v in properties.values()])
        if np.all(n_unique_properties == 1):
            with self.block_update_properties():
                self.current_properties = properties
        self._set_highlight()

    def interaction_box(self, index) -> Optional[np.ndarray]:
        """Create the interaction box around a list of points in view.

        Parameters
        ----------
        index : list
            List of points around which to construct the interaction box.

        Returns
        -------
        box : np.ndarray or None
            4x2 array of corners of the interaction box in clockwise order
            starting in the upper-left corner.
        """
        if len(index) > 0:
            data = self._view_data[index]
            size = self._view_size[index]
            data = points_to_squares(data, size)
            return create_box(data)
        return None

    @property
    def mode(self) -> str:
        """str: Interactive mode

        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In ADD mode clicks of the cursor add points at the clicked location.

        In SELECT mode the cursor can select points by clicking on them or
        by dragging a box around them. Once selected points can be moved,
        have their properties edited, or be deleted.
        """
        return str(self._mode)

    @mode.setter
    def mode(self, mode):
        mode = Mode(mode)

        if not self.editable:
            mode = Mode.PAN_ZOOM

        if mode == self._mode:
            return
        old_mode = self._mode

        if old_mode == Mode.ADD:
            self.mouse_drag_callbacks.remove(add)
        elif old_mode == Mode.SELECT:
            # add mouse drag and move callbacks
            self.mouse_drag_callbacks.remove(select)
            self.mouse_move_callbacks.remove(highlight)

        if mode == Mode.ADD:
            self.cursor = 'pointing'
            self.interactive = True
            self.help = trans._('hold <space> to pan/zoom')
            self.selected_data = set()
            self._set_highlight()
            self.mouse_drag_callbacks.append(add)
        elif mode == Mode.SELECT:
            self.cursor = 'standard'
            self.interactive = False
            self.help = trans._('hold <space> to pan/zoom')
            # add mouse drag and move callbacks
            self.mouse_drag_callbacks.append(select)
            self.mouse_move_callbacks.append(highlight)
        elif mode == Mode.PAN_ZOOM:
            self.cursor = 'standard'
            self.interactive = True
            self.help = ''
        else:
            raise ValueError(
                trans._(
                    "Mode not recognized",
                    deferred=True,
                )
            )

        if mode != Mode.SELECT or old_mode != Mode.SELECT:
            self._selected_data_stored = set()

        self._mode = mode
        self._set_highlight()

        self.events.mode(mode=mode)

    @property
    def _view_data(self) -> np.ndarray:
        """Get the coords of the points in view

        Returns
        -------
        view_data : (N x D) np.ndarray
            Array of coordinates for the N points in view
        """
        if len(self._indices_view) > 0:
            data = self.data[np.ix_(self._indices_view, self._dims_displayed)]
        else:
            # if no points in this slice send dummy data
            data = np.zeros((0, self._ndisplay))

        return data

    @property
    def _view_text(self) -> np.ndarray:
        """Get the values of the text elements in view

        Returns
        -------
        text : (N x 1) np.ndarray
            Array of text strings for the N text elements in view
        """
        return self.text.view_text(self._indices_view)

    @property
    def _view_text_coords(self) -> Tuple[np.ndarray, str, str]:
        """Get the coordinates of the text elements in view

        Returns
        -------
        text_coords : (N x D) np.ndarray
            Array of coordindates for the N text elements in view
        """
        # TODO check if it is used, as it has wrong signature and this not cause errors.
        return self.text.compute_text_coords(self._view_data, self._ndisplay)

    @property
    def _view_size(self) -> np.ndarray:
        """Get the sizes of the points in view

        Returns
        -------
        view_size : (N x D) np.ndarray
            Array of sizes for the N points in view
        """
        if len(self._indices_view) > 0:
            # Get the point sizes and scale for ndim display
            sizes = (
                self.size[
                    np.ix_(self._indices_view, self._dims_displayed)
                ].mean(axis=1)
                * self._view_size_scale
            )

        else:
            # if no points, return an empty list
            sizes = np.array([])
        return sizes

    @property
    def _view_face_color(self) -> np.ndarray:
        """Get the face colors of the points in view

        Returns
        -------
        view_face_color : (N x 4) np.ndarray
            RGBA color array for the face colors of the N points in view.
            If there are no points in view, returns array of length 0.
        """
        return self.face_color[self._indices_view]

    @property
    def _view_edge_color(self) -> np.ndarray:
        """Get the edge colors of the points in view

        Returns
        -------
        view_edge_color : (N x 4) np.ndarray
            RGBA color array for the edge colors of the N points in view.
            If there are no points in view, returns array of length 0.
        """
        return self.edge_color[self._indices_view]

    def _set_editable(self, editable=None):
        """Set editable mode based on layer properties."""
        if editable is None:
            self.editable = self._ndisplay < 3
        if not self.editable:
            self.mode = Mode.PAN_ZOOM

    def _slice_data(
        self, dims_indices
    ) -> Tuple[List[int], Union[float, np.ndarray]]:
        """Determines the slice of points given the indices.

        Parameters
        ----------
        dims_indices : sequence of int or slice
            Indices to slice with.

        Returns
        -------
        slice_indices : list
            Indices of points in the currently viewed slice.
        scale : float, (N, ) array
            If in `n_dimensional` mode then the scale factor of points, where
            values of 1 corresponds to points located in the slice, and values
            less than 1 correspond to points located in neighboring slices.
        """
        # Get a list of the data for the points in this slice
        not_disp = list(self._dims_not_displayed)
        indices = np.array(dims_indices)
        if len(self.data) > 0:
            if self.n_dimensional is True and self.ndim > 2:
                distances = abs(self.data[:, not_disp] - indices[not_disp])
                sizes = self.size[:, not_disp] / 2
                matches = np.all(distances <= sizes, axis=1)
                size_match = sizes[matches]
                size_match[size_match == 0] = 1
                scale_per_dim = (size_match - distances[matches]) / size_match
                scale_per_dim[size_match == 0] = 1
                scale = np.prod(scale_per_dim, axis=1)
                slice_indices = np.where(matches)[0].astype(int)
                return slice_indices, scale
            else:
                data = self.data[:, not_disp]
                distances = np.abs(data - indices[not_disp])
                matches = np.all(distances < 1e-5, axis=1)
                slice_indices = np.where(matches)[0].astype(int)
                return slice_indices, 1
        else:
            return [], np.empty(0)

    def _get_value(self, position) -> Union[None, int]:
        """Value of the data at a position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : int or None
            Index of point that is at the current coordinate if any.
        """
        # Display points if there are any in this slice
        view_data = self._view_data
        selection = None
        if len(view_data) > 0:
            displayed_position = [position[i] for i in self._dims_displayed]
            # Get the point sizes
            distances = abs(view_data - displayed_position)
            in_slice_matches = np.all(
                distances <= np.expand_dims(self._view_size, axis=1) / 2,
                axis=1,
            )
            indices = np.where(in_slice_matches)[0]
            if len(indices) > 0:
                selection = self._indices_view[indices[-1]]

        return selection

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""
        # get the indices of points in view
        indices, scale = self._slice_data(self._slice_indices)
        self._view_size_scale = scale
        self._indices_view = np.array(indices)
        # get the selected points that are in view
        self._selected_view = list(
            np.intersect1d(
                np.array(list(self._selected_data)),
                self._indices_view,
                return_indices=True,
            )[2]
        )
        with self.events.highlight.blocker():
            self._set_highlight(force=True)

    def _set_highlight(self, force=False):
        """Render highlights of shapes including boundaries, vertices,
        interaction boxes, and the drag selection box when appropriate.
        Highlighting only occurs in Mode.SELECT.

        Parameters
        ----------
        force : bool
            Bool that forces a redraw to occur when `True`
        """
        # Check if any point ids have changed since last call
        if (
            self.selected_data == self._selected_data_stored
            and self._value == self._value_stored
            and np.all(self._drag_box == self._drag_box_stored)
        ) and not force:
            return
        self._selected_data_stored = copy(self.selected_data)
        self._value_stored = copy(self._value)
        self._drag_box_stored = copy(self._drag_box)

        if self._value is not None or len(self._selected_view) > 0:
            if len(self._selected_view) > 0:
                index = copy(self._selected_view)
                # highlight the hovered point if not in adding mode
                if (
                    self._value in self._indices_view
                    and self._mode == Mode.SELECT
                    and not self._is_selecting
                ):
                    hover_point = list(self._indices_view).index(self._value)
                    if hover_point not in index:
                        index.append(hover_point)
                index.sort()
            else:
                # only highlight hovered points in select mode
                if (
                    self._value in self._indices_view
                    and self._mode == Mode.SELECT
                    and not self._is_selecting
                ):
                    hover_point = list(self._indices_view).index(self._value)
                    index = [hover_point]
                else:
                    index = []

            self._highlight_index = index
        else:
            self._highlight_index = []

        # only display dragging selection box in 2D
        if self._ndisplay == 2 and self._is_selecting:
            pos = create_box(self._drag_box)
            pos = pos[list(range(4)) + [0]]
        else:
            pos = None

        self._highlight_box = pos
        self.events.highlight()

    def _update_thumbnail(self):
        """Update thumbnail with current points and colors."""
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1
        view_data = self._view_data
        if len(view_data) > 0:
            de = self._extent_data
            min_vals = [de[0, i] for i in self._dims_displayed]
            shape = np.ceil(
                [de[1, i] - de[0, i] + 1 for i in self._dims_displayed]
            ).astype(int)
            zoom_factor = np.divide(
                self._thumbnail_shape[:2], shape[-2:]
            ).min()
            if len(view_data) > self._max_points_thumbnail:
                thumbnail_indices = np.random.randint(
                    0, len(view_data), self._max_points_thumbnail
                )
                points = view_data[thumbnail_indices]
            else:
                points = view_data
                thumbnail_indices = self._indices_view
            coords = np.floor(
                (points[:, -2:] - min_vals[-2:] + 0.5) * zoom_factor
            ).astype(int)
            coords = np.clip(
                coords, 0, np.subtract(self._thumbnail_shape[:2], 1)
            )
            colors = self._face.colors[thumbnail_indices]
            colormapped[coords[:, 0], coords[:, 1]] = colors

        colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    def add(self, coord):
        """Adds point at coordinate.

        Parameters
        ----------
        coord : sequence of indices to add point at
        """
        self.data = np.append(self.data, np.atleast_2d(coord), axis=0)

    def remove_selected(self):
        """Removes selected points if any."""
        index = list(self.selected_data)
        index.sort()
        if len(index):
            self._size = np.delete(self._size, index, axis=0)
            with self._edge.events.blocker_all():
                self._edge._remove(indices_to_remove=index)
            with self._face.events.blocker_all():
                self._face._remove(indices_to_remove=index)
            for k in self.properties:
                self.properties[k] = np.delete(
                    self.properties[k], index, axis=0
                )
            self.text.remove(index)
            if self._value in self.selected_data:
                self._value = None
            self.selected_data = set()
            self.data = np.delete(self.data, index, axis=0)

    def _move(self, index, coord):
        """Moves points relative drag start location.

        Parameters
        ----------
        index : list
            Integer indices of points to move
        coord : tuple
            Coordinates to move points to
        """
        if len(index) > 0:
            index = list(index)
            disp = list(self._dims_displayed)
            if self._drag_start is None:
                center = self.data[np.ix_(index, disp)].mean(axis=0)
                self._drag_start = np.array(coord)[disp] - center
            center = self.data[np.ix_(index, disp)].mean(axis=0)
            shift = np.array(coord)[disp] - center - self._drag_start
            self.data[np.ix_(index, disp)] = (
                self.data[np.ix_(index, disp)] + shift
            )
            self.refresh()

    def _paste_data(self):
        """Paste any point from clipboard and select them."""
        npoints = len(self._view_data)
        totpoints = len(self.data)

        if len(self._clipboard.keys()) > 0:
            not_disp = self._dims_not_displayed
            data = deepcopy(self._clipboard['data'])
            offset = [
                self._slice_indices[i] - self._clipboard['indices'][i]
                for i in not_disp
            ]
            data[:, not_disp] = data[:, not_disp] + np.array(offset)
            self._data = np.append(self.data, data, axis=0)
            self._size = np.append(
                self.size, deepcopy(self._clipboard['size']), axis=0
            )
            self._edge._paste(
                colors=self._clipboard['edge_color'],
                properties=self._clipboard['properties'],
            )
            self._face._paste(
                colors=self._clipboard['face_color'],
                properties=self._clipboard['properties'],
            )

            for k in self.properties:
                self.properties[k] = np.concatenate(
                    (self.properties[k], self._clipboard['properties'][k]),
                    axis=0,
                )
            self._selected_view = list(
                range(npoints, npoints + len(self._clipboard['data']))
            )
            self._selected_data = set(
                range(totpoints, totpoints + len(self._clipboard['data']))
            )

            if len(self._clipboard['text']) > 0:
                self.text._values = np.concatenate(
                    (self.text.values, self._clipboard['text']), axis=0
                )

            self.refresh()

    def _copy_data(self):
        """Copy selected points to clipboard."""
        if len(self.selected_data) > 0:
            index = list(self.selected_data)
            self._clipboard = {
                'data': deepcopy(self.data[index]),
                'edge_color': deepcopy(self.edge_color[index]),
                'face_color': deepcopy(self.face_color[index]),
                'size': deepcopy(self.size[index]),
                'properties': {
                    k: deepcopy(v[index]) for k, v in self.properties.items()
                },
                'indices': self._slice_indices,
            }

            if len(self.text.values) == 0:
                self._clipboard['text'] = np.empty(0)

            else:
                self._clipboard['text'] = deepcopy(self.text.values[index])

        else:
            self._clipboard = {}
