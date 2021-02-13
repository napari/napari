import warnings
from copy import copy
from typing import Dict, Tuple, Union

import numpy as np

from ...utils.colormaps import Colormap, ValidColormapArg, ensure_colormap
from ...utils.events import Event
from ..base import Layer
from ..utils.color_transformations import (
    normalize_and_broadcast_colors,
    transform_color_cycle,
    transform_color_with_defaults,
)
from ..utils.layer_utils import (
    dataframe_to_properties,
    guess_continuous,
    map_property,
)
from ._vector_utils import generate_vector_meshes, vectors_to_coordinates
from ._vectors_constants import DEFAULT_COLOR_CYCLE, ColorMode


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
        Color of all of the vectors.
    edge_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to edge_color if a
        categorical attribute is used color the vectors.
    edge_colormap : str, napari.utils.Colormap
        Colormap to set vector color if a continuous attribute is used to set edge_color.
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
        Color of all of the vectors.
    edge_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to edge_color if a
        categorical attribute is used color the vectors.
    edge_colormap : str, napari.utils.Colormap
        Colormap to set vector color if a continuous attribute is used to set edge_color.
    edge_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())

    Extended Summary
    ----------
    _view_data : (M, 2, 2) array
        The start point and projections of N vectors in 2D for vectors whose
        start point is in the currently viewed slice.
    _view_face_color : (M, 4) np.ndarray
        colors for the M in view vectors
    _view_indices : (1, M) array
        indices for the M in view vectors
    _view_vertices : (4M, 2) or (8M, 2) np.ndarray
        the corner points for the M in view faces. Shape is (4M, 2) for 2D and (8M, 2) for 3D.
    _view_faces : (2M, 3) or (4M, 3) np.ndarray
        indices of the _mesh_vertices that form the faces of the M in view vectors.
        Shape is (2M, 2) for 2D and (4M, 2) for 3D.
    _property_choices : dict {str: array (N,)}
        Possible values for the properties in Vectors.properties.
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
        rotate=None,
        shear=None,
        affine=None,
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
            rotate=rotate,
            shear=shear,
            affine=affine,
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
            properties=Event,
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
            properties, _ = dataframe_to_properties(properties)
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
            self._edge_color_mode = ColorMode.DIRECT
            self.edge_color = edge_color
            if edge_color_cycle is None:
                edge_color_cycle = DEFAULT_COLOR_CYCLE
            self.edge_color_cycle = edge_color_cycle
            self.edge_color_cycle_map = {}
            self.edge_colormap = edge_colormap
            self._edge_contrast_limits = edge_contrast_limits

        self.refresh_colors()

        # Data containing vectors in the currently viewed slice
        self._view_data = np.empty((0, 2, 2))
        self._displayed_stored = []
        self._view_vertices = []
        self._view_faces = []
        self._view_indices = []

        # now that everything is set up, make the layer visible (if set to visible)
        self.visible = visible

    @property
    def data(self) -> np.ndarray:
        """(N, 2, D) array: start point and projections of vectors."""
        return self._data

    @data.setter
    def data(self, vectors: np.ndarray):
        self._data = vectors_to_coordinates(vectors)

        vertices, triangles = generate_vector_meshes(
            self._data[:, :, list(self._dims_displayed)],
            self.edge_width,
            self.length,
        )
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles
        self._displayed_stored = copy(self._dims_displayed)

        self._update_dims()
        self.events.data(value=self.data)
        self._set_editable()

    @property
    def properties(self) -> Dict[str, np.ndarray]:
        """dict {str: array (N,)}, DataFrame: Annotations for each point"""
        return self._properties

    @properties.setter
    def properties(self, properties: Dict[str, np.ndarray]):
        if not isinstance(properties, dict):
            properties, _ = dataframe_to_properties(properties)
        self._properties = self._validate_properties(properties)
        if self._edge_color_property and (
            self._edge_color_property not in self._properties
        ):
            self._edge_color_property = ''
            warnings.warn('property used for edge_color dropped')
        self.events.properties()

    def _validate_properties(
        self, properties: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
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
                'edge_colormap': self.edge_colormap.name,
                'edge_contrast_limits': self.edge_contrast_limits,
                'data': self.data,
                'properties': self.properties,
            }
        )
        return state

    def _get_ndim(self) -> int:
        """Determine number of dimensions of the layer."""
        return self.data.shape[2]

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
            # Convert from projections to endpoints using the current length
            data = copy(self.data)
            data[:, 1, :] = data[:, 0, :] + self.length * data[:, 1, :]
            maxs = np.max(data, axis=(0, 1))
            mins = np.min(data, axis=(0, 1))
            extrema = np.vstack([mins, maxs])
        return extrema

    @property
    def edge_width(self) -> Union[int, float]:
        """float: Width for all vectors in pixels."""
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: Union[int, float]):
        self._edge_width = edge_width

        vertices, triangles = generate_vector_meshes(
            self.data[:, :, list(self._dims_displayed)],
            self._edge_width,
            self.length,
        )
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles
        self._displayed_stored = copy(self._dims_displayed)

        self.events.edge_width()
        self.refresh()

    @property
    def length(self) -> Union[int, float]:
        """float: Multiplicative factor for length of all vectors."""
        return self._length

    @length.setter
    def length(self, length: Union[int, float]):
        self._length = length

        vertices, triangles = generate_vector_meshes(
            self.data[:, :, list(self._dims_displayed)],
            self.edge_width,
            self._length,
        )
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles
        self._displayed_stored = copy(self._dims_displayed)

        self.events.length()
        self.refresh()

    @property
    def edge_color(self) -> np.ndarray:
        """(1 x 4) np.ndarray: Array of RGBA edge colors (applied to all vectors)"""
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: str):
        # save the old mode, we will emit an event if the mode has changed
        old_mode = self._edge_color_mode

        # if the provided face color is a string, first check if it is a key in the properties.
        # otherwise, assume it is the name of a color
        if self._is_color_mapped(edge_color):
            if guess_continuous(self.properties[edge_color]):
                new_mode = ColorMode.COLORMAP
                self._edge_color_mode = new_mode
            else:
                new_mode = ColorMode.CYCLE
                self._edge_color_mode = new_mode
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
            new_mode = ColorMode.DIRECT
            self._edge_color_mode = new_mode
            self._edge_color_property = ''

            self.events.edge_color()

            if self.visible:
                self._update_thumbnail()
        if new_mode != old_mode:
            self.events.edge_color_mode()

    def refresh_colors(self, update_color_mapping: bool = False):
        """Calculate and update edge colors if using a cycle or color map

        Parameters
        ----------
        update_color_mapping : bool
            If set to True, the function will recalculate the color cycle map
            or colormap (whichever is being used). If set to False, the function
            will use the current color cycle map or color map. For example, if you
            are adding/modifying vectors and want them to be colored with the same
            mapping as the other vectors (i.e., the new vectors shouldn't affect
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
                            self._edge_color_cycle,
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
                                self._edge_color_cycle
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
                            colormap=self.edge_colormap,
                        )
                        self.edge_contrast_limits = contrast_limits
                    else:
                        edge_colors, _ = map_property(
                            prop=edge_color_properties,
                            colormap=self.edge_colormap,
                            contrast_limits=self.edge_contrast_limits,
                        )
                else:
                    edge_colors = np.empty((0, 4))
                self._edge_color = edge_colors
            self.events.edge_color()
            if self.visible:
                self._update_thumbnail()

    def _is_color_mapped(self, color) -> bool:
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
                'edge_color should be the name of a color, an array of colors, or the name of an property'
            )

    @property
    def edge_color_mode(self) -> ColorMode:
        """str: Edge color setting mode

        DIRECT (default mode) allows each vector to be set arbitrarily

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
    def edge_color_cycle(self) -> np.ndarray:
        """list, np.ndarray :  Color cycle for edge_color.
        Can be a list of colors defined by name, RGB or RGBA
        """
        return self._edge_color_cycle_values

    @edge_color_cycle.setter
    def edge_color_cycle(self, edge_color_cycle: Union[list, np.ndarray]):
        transformed_color_cycle, transformed_colors = transform_color_cycle(
            color_cycle=edge_color_cycle,
            elem_name='edge_color_cycle',
            default="white",
        )
        self._edge_color_cycle_values = transformed_colors
        self._edge_color_cycle = transformed_color_cycle
        if self._edge_color_mode == ColorMode.CYCLE:
            self.refresh_colors(update_color_mapping=True)

    @property
    def edge_colormap(self) -> Tuple[str, Colormap]:
        """Return the colormap to be applied to a property to get the edge color.

        Returns
        -------
        colormap : napari.utils.Colormap
            The Colormap object.
        """
        return self._edge_colormap

    @edge_colormap.setter
    def edge_colormap(self, colormap: ValidColormapArg):
        self._edge_colormap = ensure_colormap(colormap)

    @property
    def edge_contrast_limits(self) -> Tuple[float, float]:
        """None, (float, float): contrast limits for mapping
        the edge_color colormap property to 0 and 1
        """
        return self._edge_contrast_limits

    @edge_contrast_limits.setter
    def edge_contrast_limits(
        self, contrast_limits: Union[None, Tuple[float, float]]
    ):
        self._edge_contrast_limits = contrast_limits

    @property
    def _view_face_color(self) -> np.ndarray:
        """" (Mx4) np.ndarray : colors for the M in view vectors"""
        face_color = np.repeat(self.edge_color[self._view_indices], 2, axis=0)
        if self._ndisplay == 3 and self.ndim > 2:
            face_color = np.vstack([face_color, face_color])

        return face_color

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""

        if not self._dims_displayed == self._displayed_stored:
            vertices, triangles = generate_vector_meshes(
                self.data[:, :, list(self._dims_displayed)],
                self.edge_width,
                self.length,
            )
            self._mesh_vertices = vertices
            self._mesh_triangles = triangles
            self._displayed_stored = copy(self._dims_displayed)

        vertices = self._mesh_vertices
        not_disp = list(self._dims_not_displayed)
        disp = list(self._dims_displayed)
        indices = np.array(self._slice_indices)

        if len(self.data) == 0:
            faces = []
            self._view_data = np.empty((0, 2, 2))
            self._view_indices = []
        elif self.ndim > 2:
            data = self.data[:, 0, not_disp].astype('int')
            matches = np.all(data == indices[not_disp], axis=1)
            matches = np.where(matches)[0]
            self._view_indices = matches
            self._view_data = self.data[np.ix_(matches, [0, 1], disp)]
            if len(matches) == 0:
                faces = []
            else:
                keep_inds = np.repeat(2 * matches, 2)
                keep_inds[1::2] = keep_inds[1::2] + 1
                if self._ndisplay == 3:
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
            self._view_data = self.data[:, :, disp]
            self._view_indices = np.arange(self.data.shape[0])

        if len(faces) == 0:
            self._view_vertices = []
            self._view_faces = []
        else:
            self._view_vertices = vertices
            self._view_faces = faces

    def _update_thumbnail(self):
        """Update thumbnail with current vectors and colors."""
        # calculate min vals for the vertices and pad with 0.5
        # the offset is needed to ensure that the top left corner of the
        # vectors corresponds to the top left corner of the thumbnail
        de = self._extent_data
        offset = (np.array([de[0, d] for d in self._dims_displayed]) + 0.5)[
            -2:
        ]
        # calculate range of values for the vertices and pad with 1
        # padding ensures the entire vector can be represented in the thumbnail
        # without getting clipped
        shape = np.ceil(
            [de[1, d] - de[0, d] + 1 for d in self._dims_displayed]
        ).astype(int)[-2:]
        zoom_factor = np.divide(self._thumbnail_shape[:2], shape).min()

        # vectors = copy(self._data_view[:, :, -2:])
        if self._view_data.shape[0] > self._max_vectors_thumbnail:
            thumbnail_indices = np.random.randint(
                0, self._view_data.shape[0], self._max_vectors_thumbnail
            )
            vectors = copy(self._view_data[thumbnail_indices, :, -2:])
            thumbnail_color_indices = self._view_indices[thumbnail_indices]
        else:
            vectors = copy(self._view_data[:, :, -2:])
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

    def _get_value(self, position):
        """Value of the data at a position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : None
            Value of the data at the coord.
        """
        return None
