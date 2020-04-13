from itertools import cycle
from typing import Union, Dict, Tuple
from xml.etree.ElementTree import Element
import warnings

import numpy as np
from copy import copy
from ..base import Layer
from ._vectors_constants import ColorMode, DEFAULT_COLOR_CYCLE
from ..utils.color_transformations import (
    transform_color_with_defaults,
    transform_color_cycle,
    normalize_and_broadcast_colors,
)
from ..utils.layer_utils import (
    dataframe_to_properties,
    guess_continuous,
    map_property,
)
from ...utils.event import Event
from ...utils.status_messages import format_float
from ...utils.colormaps.standardize_color import transform_color
from ._vector_utils import vectors_to_coordinates, generate_vector_meshes
from vispy.color import get_colormap
from vispy.color.colormap import Colormap


class Vectors(Layer):
    """
    Vectors layer renders lines onto the canvas.

    Parameters
    ----------
    data : (N, 2, D) or (N1, N2, ..., ND, D) array
        An (N, 2, D) array is interpreted as "coordinate-like" data and a
        list of N vectors with start point and projections of the vector in
        D dimensions. An (N1, N2, ..., ND, D) array is interpreted as
        "image-like" data where there is a length D vector of the
        projections at each pixel.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each vector. Each property should be an array of length N,
        where N is the number of vectors.
    edge_width : float
        Width for all vectors in pixels.
    length : float
         Multiplicative factor on projections for length of all vectors.
    edge_color : str
        Edge color of all the vectors.
    edge_color_cycle : np.ndarray, list, cycle
        Cycle of colors (provided as RGBA) to map to edge_color if a
        categorical attribute is used to set face_color.
    edge_colormap : str, vispy.color.colormap.Colormap
        Colormap to set edge_color if a continuous attribute is used to set edge_color.
        See vispy docs for details: http://vispy.org/color.html#vispy.color.Colormap
    edge_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.

    Attributes
    ----------
    data : (N, 2, D) array
        The start point and projections of N vectors in D dimensions.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each vector. Each property should be an array of length N,
        where N is the number of vectors.
    edge_width : float
        Width for all vectors in pixels.
    length : float
         Multiplicative factor on projections for length of all vectors.
    edge_color : str
        Edge color of all the vectors.
    edge_color_cycle : np.ndarray, list, cycle
        Cycle of colors (provided as RGBA) to map to edge_color if a
        categorical attribute is used to set face_color.
    edge_colormap : str, vispy.color.colormap.Colormap
        Colormap to set edge_color if a continuous attribute is used to set face_color.
        See vispy docs for details: http://vispy.org/color.html#vispy.color.Colormap
    edge_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())

    Extended Summary
    ----------
    _data_view : (M, 2, 2) array
        The start point and projections of N vectors in 2D for vectors whose
        start point is in the currently viewed slice.
    _property_choices : dict {str: array (N,)}
        Possible values for the properties in Points.properties.
        If properties is not provided, it will be {} (empty dictionary).
    _mesh_vertices : (4N, 2) array
        The four corner points for the mesh representation of each vector as as
        rectangle in the slice that it starts in.
    _mesh_triangles : (2N, 3) array
        The integer indices of the `_mesh_vertices` that form the two triangles
        for the mesh representation of the vectors.
    _max_vectors_thumbnail : int
        The maximum number of vectors that will ever be used to render the
        thumbnail. If more vectors are present then they are randomly
        subsampled.
    """

    # The max number of vectors that will ever be used to render the thumbnail
    # If more vectors are present then they are randomly subsampled
    _max_vectors_thumbnail = 1024

    def __init__(
        self,
        data,
        *,
        properties=None,
        edge_width=1,
        edge_color='red',
        edge_color_cycle=None,
        edge_colormap='viridis',
        edge_contrast_limits=None,
        length=1,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        opacity=0.7,
        blending='translucent',
        visible=True,
    ):

        super().__init__(
            data,
            2,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            opacity=opacity,
            blending=blending,
            visible=visible,
        )

        # events for non-napari calculations
        self.events.add(
            length=Event,
            edge_width=Event,
            edge_color=Event,
            edge_color_mode=Event,
            current_edge_color=Event,
        )

        self.visible = False

        # Save the vector style params
        self._edge_width = edge_width

        # length attribute
        self._length = length

        self.data = data

        # Save the properties
        if properties is None:
            self._properties = {}
            self._property_choices = {}
        elif len(data) > 0:
            properties = dataframe_to_properties(properties)
            self._properties = self._validate_properties(properties)
            self._property_choices = {
                k: np.unique(v) for k, v in properties.items()
            }
        elif len(data) == 0:
            self._property_choices = {
                k: np.asarray(v) for k, v in properties.items()
            }
            empty_properties = {
                k: np.empty(0, dtype=v.dtype)
                for k, v in self._property_choices.items()
            }
            self._properties = empty_properties

        with self.block_update_properties():
            self._edge_color_property = ''
            self.edge_color = edge_color
            if edge_color_cycle is None:
                edge_color_cycle = DEFAULT_COLOR_CYCLE
            self.edge_color_cycle = edge_color_cycle
            self.edge_color_cycle_map = {}
            self.edge_colormap = edge_colormap
            self._edge_contrast_limits = edge_contrast_limits

        self.refresh_colors()

        # set the current_* properties
        if len(data) > 0:
            self._current_edge_color = self.edge_color[-1]
        elif len(data) == 0 and self.properties:
            if self._edge_color_mode == ColorMode.DIRECT:
                self._current_edge_color = transform_color_with_defaults(
                    num_entries=1,
                    colors=edge_color,
                    elem_name="edge_color",
                    default="white",
                )
            elif self._edge_color_mode == ColorMode.CYCLE:
                curr_edge_color = transform_color(next(self.edge_color_cycle))
                prop_value = self._property_choices[self._edge_color_property][
                    0
                ]
                self.edge_color_cycle_map[prop_value] = curr_edge_color
                self._current_edge_color = curr_edge_color
            elif self._edge_color_mode == ColorMode.COLORMAP:
                prop_value = self._property_choices[self._edge_color_property][
                    0
                ]
                curr_edge_color, _ = map_property(
                    prop=prop_value,
                    colormap=self.edge_colormap[1],
                    contrast_limits=self._edge_contrast_limits,
                )
                self._current_edge_color = curr_edge_color

        else:
            self._current_edge_color = self.edge_color[-1]
            self.current_properties = {}

        # self._mesh_vertices = np.empty((0, 2))
        # self._mesh_triangles = np.empty((0, 3), dtype=np.uint32)

        # Data containing vectors in the currently viewed slice
        self._data_view = np.empty((0, 2, 2))
        self._displayed_stored = []
        self._view_vertices = []
        self._view_faces = []
        self._view_indices = []

        # now that everything is set up, make the layer visible (if set to visible)
        self.visible = visible

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, vectors: np.ndarray):
        """(N, 2, D) array: start point and projections of vectors."""

        self._data = vectors_to_coordinates(vectors)

        vertices, triangles = generate_vector_meshes(
            self._data[:, :, list(self.dims.displayed)],
            self.edge_width,
            self.length,
        )
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles
        self._displayed_stored = copy(self.dims.displayed)

        self._update_dims()
        self.events.data()

    @property
    def properties(self):
        """dict {str: array (N,)}, DataFrame: Annotations for each point"""
        return self._properties

    @properties.setter
    def properties(self, properties: Dict[str, np.ndarray]):
        if not isinstance(properties, dict):
            properties = dataframe_to_properties(properties)
        self._properties = self._validate_properties(properties)
        if self._edge_color_property and (
            self._edge_color_property not in self._properties
        ):
            self._edge_color_property = ''
            warnings.warn('property used for edge_color dropped')

    def _validate_properties(self, properties: Dict[str, np.ndarray]):
        """Validates the type and size of the properties"""
        for v in properties.values():
            if len(v) != len(self.data):
                raise ValueError(
                    'the number of properties must equal the number of points'
                )

        return properties

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
                'length': self.length,
                'edge_width': self.edge_width,
                'edge_color': self.edge_color,
                'edge_color_cycle': self.edge_color_cycle,
                'edge_colormap': self.edge_colormap[0],
                'edge_contrast_limits': self.edge_contrast_limits,
                'data': self.data,
                'properties': self.properties,
            }
        )
        return state

    def _get_ndim(self):
        """Determine number of dimensions of the layer."""
        return self.data.shape[2]

    def _get_extent(self):
        """Determine ranges for slicing given by (min, max, step)."""
        if len(self.data) == 0:
            maxs = np.ones(self.data.shape[2], dtype=int)
            mins = np.zeros(self.data.shape[2], dtype=int)
        else:
            # Convert from projections to endpoints using the current length
            data = copy(self.data)
            data[:, 1, :] = data[:, 0, :] + self.length * data[:, 1, :]
            maxs = np.max(data, axis=(0, 1))
            mins = np.min(data, axis=(0, 1))

        return [(min, max, 1) for min, max in zip(mins, maxs)]

    @property
    def edge_width(self) -> Union[int, float]:
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: Union[int, float]):
        """float: Width for all vectors in pixels."""
        self._edge_width = edge_width

        vertices, triangles = generate_vector_meshes(
            self.data[:, :, list(self.dims.displayed)],
            self._edge_width,
            self.length,
        )
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles
        self._displayed_stored = copy(self.dims.displayed)

        self.events.edge_width()
        self.refresh()
        self.status = format_float(self.edge_width)

    @property
    def length(self) -> Union[int, float]:
        return self._length

    @length.setter
    def length(self, length: Union[int, float]):
        """float: Multiplicative factor for length of all vectors."""
        self._length = length

        vertices, triangles = generate_vector_meshes(
            self.data[:, :, list(self.dims.displayed)],
            self.edge_width,
            self._length,
        )
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles
        self._displayed_stored = copy(self.dims.displayed)

        self.events.length()
        self.refresh()
        self.status = format_float(self.length)

    @property
    def edge_color(self) -> str:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: str):
        """(1 x 4) np.ndarray: Array of RGBA edge colors (applied to all vectors)"""
        # if the provided face color is a string, first check if it is a key in the properties.
        # otherwise, assume it is the name of a color
        if self._is_color_mapped(edge_color):
            if guess_continuous(self.properties[edge_color]):
                self._edge_color_mode = ColorMode.COLORMAP
            else:
                self._edge_color_mode = ColorMode.CYCLE
            self._edge_color_property = edge_color
            self.refresh_colors()

        else:
            transformed_color = transform_color_with_defaults(
                num_entries=len(self.data),
                colors=edge_color,
                elem_name="edge_color",
                default="white",
            )
            self._edge_color = normalize_and_broadcast_colors(
                len(self.data), transformed_color
            )
            self._edge_color_mode = ColorMode.DIRECT
            self._edge_color_property = ''

            self.events.edge_color()

            if self.visible:
                self._update_thumbnail()

    def refresh_colors(self, update_color_mapping: bool = False):
        """Calculate and update edge colors if using a cycle or color map

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
        if self._update_properties:
            if self._edge_color_mode == ColorMode.CYCLE:
                edge_color_properties = self.properties[
                    self._edge_color_property
                ]
                if update_color_mapping:
                    self.edge_color_cycle_map = {
                        k: c
                        for k, c in zip(
                            np.unique(edge_color_properties),
                            self.edge_color_cycle,
                        )
                    }
                else:
                    # add properties if they are not in the colormap
                    # and update_color_mapping==False
                    edge_color_cycle_keys = [*self.edge_color_cycle_map]
                    props_in_map = np.in1d(
                        edge_color_properties, edge_color_cycle_keys
                    )
                    if not np.all(props_in_map):
                        props_to_add = np.unique(
                            edge_color_properties[np.logical_not(props_in_map)]
                        )
                        for prop in props_to_add:
                            self.edge_color_cycle_map[prop] = next(
                                self.edge_color_cycle
                            )
                edge_colors = np.array(
                    [
                        self.edge_color_cycle_map[x]
                        for x in edge_color_properties
                    ]
                )
                if len(edge_colors) == 0:
                    edge_colors = np.empty((0, 4))
                self._edge_color = edge_colors
            elif self._edge_color_mode == ColorMode.COLORMAP:
                edge_color_properties = self.properties[
                    self._edge_color_property
                ]
                if len(edge_color_properties) > 0:
                    if (
                        update_color_mapping
                        or self.edge_contrast_limits is None
                    ):
                        edge_colors, contrast_limits = map_property(
                            prop=edge_color_properties,
                            colormap=self.edge_colormap[1],
                        )
                        self.edge_contrast_limits = contrast_limits
                    else:
                        edge_colors, _ = map_property(
                            prop=edge_color_properties,
                            colormap=self.edge_colormap[1],
                            contrast_limits=self.edge_contrast_limits,
                        )
                else:
                    edge_colors = np.empty((0, 4))
                self._edge_color = edge_colors
            self.events.edge_color()
            if self.visible:
                self._update_thumbnail()

    def _is_color_mapped(self, color):
        """ determines if the new color argument is for directly setting or cycle/colormap"""
        if isinstance(color, str):
            if color in self.properties:
                return True
            else:
                return False
        elif isinstance(color, (list, np.ndarray)):
            return False
        else:
            raise ValueError(
                'face_color should be the name of a color, an array of colors, or the name of an property'
            )

    @property
    def edge_color_mode(self):
        """str: Edge color setting mode

        DIRECT (default mode) allows each point to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
        """
        return str(self._edge_color_mode)

    @edge_color_mode.setter
    def edge_color_mode(self, edge_color_mode: Union[str, ColorMode]):
        edge_color_mode = ColorMode(edge_color_mode)

        if edge_color_mode == ColorMode.DIRECT:
            self._edge_color_mode = edge_color_mode
        elif edge_color_mode in (ColorMode.CYCLE, ColorMode.COLORMAP):
            if self._edge_color_property == '':
                if self.properties:
                    self._edge_color_property = next(iter(self.properties))
                    warning_msg = (
                        'edge_color_property was not set, setting to: %s'
                        % self._edge_color_property
                    )
                    warnings.warn(warning_msg, RuntimeWarning)
                else:
                    raise ValueError(
                        'There must valid properties to use %s color mode'
                        % str(edge_color_mode)
                    )
            # ColorMode.COLORMAP can only be applied to numeric properties
            if (edge_color_mode == ColorMode.COLORMAP) and not issubclass(
                self.properties[self._edge_color_property].dtype.type,
                np.number,
            ):
                raise TypeError(
                    'selected property must be numeric to use ColorMode.COLORMAP'
                )

            self._edge_color_mode = edge_color_mode
            self.refresh_colors()
        self.events.edge_color_mode()

    @property
    def edge_color_cycle(self):
        """Union[list, np.ndarray, cycle] :  Color cycle for edge_color.
        Can be a list of colors or a cycle of colors

        """
        return self._edge_color_cycle

    @edge_color_cycle.setter
    def edge_color_cycle(
        self, edge_color_cycle: Union[list, np.ndarray, cycle]
    ):
        self._edge_color_cycle = transform_color_cycle(
            color_cycle=edge_color_cycle,
            elem_name="edge_color_cycle",
            default="white",
        )
        if self._edge_color_mode == ColorMode.CYCLE:
            self.refresh_colors(update_color_mapping=True)

    @property
    def edge_colormap(self):
        """Return the colormap to be applied to a property to get the edge color.

        Returns
        -------
        colormap_name : str
            The name of the current colormap.
        colormap : vispy.color.Colormap
            The vispy colormap object.
        """
        return self._edge_colormap_name, self._edge_colormap

    @edge_colormap.setter
    def edge_colormap(self, colormap: Union[str, Colormap]):
        self._edge_colormap = get_colormap(colormap)
        if isinstance(colormap, str):
            self._edge_colormap_name = colormap
        else:
            self._edge_colormap_name = 'unknown_colormap'

    @property
    def edge_contrast_limits(self):
        """ None, (float, float): contrast limits for mapping
        the edge_color colormap property to 0 and 1
        """
        return self._edge_contrast_limits

    @edge_contrast_limits.setter
    def edge_contrast_limits(
        self, contrast_limits: Union[None, Tuple[float, float]]
    ):
        self._edge_contrast_limits = contrast_limits

    @property
    def current_edge_color(self):
        return self._current_edge_color

    @current_edge_color.setter
    def current_edge_color(self, edge_color: np.ndarray):
        self._current_edge_color = transform_color(edge_color)
        self.edge_color = self._current_edge_color
        self.events.current_edge_color()

    @property
    def _view_face_color(self) -> np.ndarray:

        face_color = np.repeat(self.edge_color[self._view_indices], 2, axis=0)
        if self.dims.ndisplay == 3 and self.ndim > 2:
            face_color = np.vstack([face_color, face_color])

        return face_color

    @property
    def _view_vertex_color(self) -> np.ndarray:

        if self.dims.ndisplay == 2:
            vertex_color = np.repeat(self.edge_color, 4, axis=0)
        elif self.dims.ndisplay == 3:
            vertex_color = np.repeat(self.edge_color, 8, axis=0)

        return vertex_color

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""

        if not self.dims.displayed == self._displayed_stored:
            vertices, triangles = generate_vector_meshes(
                self.data[:, :, list(self.dims.displayed)],
                self.edge_width,
                self.length,
            )
            self._mesh_vertices = vertices
            self._mesh_triangles = triangles
            self._displayed_stored = copy(self.dims.displayed)

        vertices = self._mesh_vertices
        not_disp = list(self.dims.not_displayed)
        disp = list(self.dims.displayed)
        indices = np.array(self.dims.indices)

        if len(self.data) == 0:
            faces = []
            self._data_view = np.empty((0, 2, 2))
            self._view_indices = []
        elif self.ndim > 2:
            data = self.data[:, 0, not_disp].astype('int')
            matches = np.all(data == indices[not_disp], axis=1)
            matches = np.where(matches)[0]
            self._view_indices = matches
            self._data_view = self.data[np.ix_(matches, [0, 1], disp)]
            if len(matches) == 0:
                faces = []
            else:
                keep_inds = np.repeat(2 * matches, 2)
                keep_inds[1::2] = keep_inds[1::2] + 1
                if self.dims.ndisplay == 3:
                    keep_inds = np.concatenate(
                        [
                            keep_inds,
                            len(self._mesh_triangles) // 2 + keep_inds,
                        ],
                        axis=0,
                    )
                faces = self._mesh_triangles[keep_inds]
        else:
            faces = self._mesh_triangles
            self._data_view = self.data[:, :, disp]
            self._view_indices = np.arange(self.data.shape[0])

        if len(faces) == 0:
            self._view_vertices = []
            self._view_faces = []
        else:
            self._view_vertices = vertices
            self._view_faces = faces

    def _update_thumbnail(self):
        """Update thumbnail with current points and colors."""
        # calculate min vals for the vertices and pad with 0.5
        # the offset is needed to ensure that the top left corner of the
        # vectors corresponds to the top left corner of the thumbnail
        offset = (
            np.array([self.dims.range[d][0] for d in self.dims.displayed])
            + 0.5
        )[-2:]
        # calculate range of values for the vertices and pad with 1
        # padding ensures the entire vector can be represented in the thumbnail
        # without getting clipped
        shape = np.ceil(
            [
                self.dims.range[d][1] - self.dims.range[d][0] + 1
                for d in self.dims.displayed
            ]
        ).astype(int)[-2:]
        zoom_factor = np.divide(self._thumbnail_shape[:2], shape).min()

        # vectors = copy(self._data_view[:, :, -2:])
        if self._data_view.shape[0] > self._max_vectors_thumbnail:
            thumbnail_indices = np.random.randint(
                0, self._data_view.shape[0], self._max_vectors_thumbnail
            )
            vectors = copy(self._data_view[thumbnail_indices, :, -2:])
            thumbnail_color_indices = self._view_indices[thumbnail_indices]
        else:
            vectors = copy(self._data_view[:, :, -2:])
            thumbnail_color_indices = self._view_indices
        vectors[:, 1, :] = vectors[:, 0, :] + vectors[:, 1, :] * self.length
        downsampled = (vectors - offset) * zoom_factor
        downsampled = np.clip(
            downsampled, 0, np.subtract(self._thumbnail_shape[:2], 1)
        )
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1
        edge_colors = self.edge_color[thumbnail_color_indices]
        for v, ec in zip(downsampled, edge_colors):
            start = v[0]
            stop = v[1]
            step = int(np.ceil(np.max(abs(stop - start))))
            x_vals = np.linspace(start[0], stop[0], step)
            y_vals = np.linspace(start[1], stop[1], step)
            for x, y in zip(x_vals, y_vals):
                colormapped[int(x), int(y), :] = ec
        colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    def to_xml_list(self):
        """Convert vectors to a list of svg xml elements.

        Returns
        ----------
        xml : list
            List of xml elements defining each vector as a line according to
            the svg specification.
        """
        xml_list = []

        width = str(self.edge_width)
        edge_color = self.edge_color[self._view_indices, :3]

        opacity = str(self.opacity)
        props = {'stroke-width': width, 'opacity': opacity}

        for v, ec in zip(self._data_view, edge_color):
            x1 = str(v[0, -2])
            y1 = str(v[0, -1])
            x2 = str(v[0, -2] + self.length * v[1, -2])
            y2 = str(v[0, -1] + self.length * v[1, -1])
            stroke = f'rgb{tuple(ec)}'
            props['stroke'] = stroke
            element = Element('line', x1=y1, y1=x1, x2=y2, y2=x2, **props)
            xml_list.append(element)

        return xml_list

    def _get_value(self):
        """Returns coordinates, values, and a string for a given mouse position
        and set of indices.

        Returns
        ----------
        value : int, None
            Value of the data at the coord.
        """

        return None
