from typing import Union
from xml.etree.ElementTree import Element
import numpy as np
from copy import copy
from ..base import Layer
from vispy.scene.visuals import Mesh
from ...util.event import Event
from .vectors_util import vectors_to_coordinates, generate_vector_meshes
from vispy.color import get_color_names, Color


class Vectors(Layer):
    """
    Vectors layer renders lines onto the canvas.

    Parameters
    ----------
    vectors : (N, 2, D) or (N1, N2, ..., ND, D) array
        An (N, 2, D) array is interpreted as "coordinate-like" data and a
        list of N vectors with start point and projections of the vector in
        D dimensions. An (N1, N2, ..., ND, D) array is interpreted as
        "image-like" data where there is a length D vector of the
        projections at each pixel.
    edge_width : float
        Width for all vectors in pixels.
    length : float
         Multiplicative factor on projections for length of all vectors.
    edge_color : str
        Edge color of all the vectors.
    name : str
        Name of the layer.

    Attributes
    ----------
    data : (N, 2, D) array
        The start point and projections of N vectors in D dimensions.
    edge_width : float
        Width for all vectors in pixels.
    length : float
         Multiplicative factor on projections for length of all vectors.
    edge_color : str
        Edge color of all the vectors.

    Extended Summary
    ----------
    _data_view : (M, 2, 2) array
        The start point and projections of N vectors in 2D for vectors whose
        start point is in the currently viewed slice.
    _mesh_vertices : (4N, 2) array
        The four corner points for the mesh representation of each vector as as
        rectangle in the slice that it starts in.
    _mesh_vertices : (2N, 3) array
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

    class_keymap = {}

    def __init__(
        self, vectors, *, edge_width=1, edge_color='red', length=1, name=None
    ):

        super().__init__(Mesh(), name)

        # events for non-napari calculations
        self.events.add(length=Event, edge_width=Event)

        # Save the vector style params
        self._edge_width = edge_width
        self._edge_color = edge_color
        self._colors = get_color_names()

        self._mesh_vertices = np.empty((0, 2))
        self._mesh_triangles = np.empty((0, 3), dtype=np.uint32)

        # Data containing vectors in the currently viewed slice
        self._data_view = np.empty((0, 2, 2))

        # length attribute
        self._length = length

        # update flags
        self._need_display_update = False
        self._need_visual_update = False

        # assign vector data and establish default behavior
        with self.freeze_refresh():
            self.data = vectors

            # Re intitialize indices depending on image dims
            self._indices = (0,) * (self.ndim - 2) + (
                slice(None, None, None),
                slice(None, None, None),
            )

            # Trigger generation of view slice and thumbnail
            self._set_view_slice()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, vectors: np.ndarray):
        """(N, 2, D) array: start point and projections of vectors."""

        self._data = vectors_to_coordinates(vectors)

        vertices, triangles = generate_vector_meshes(
            self._data, self.edge_width, self.length
        )
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles

        self.events.data()
        self.refresh()

    def _get_shape(self):
        return [r[1] for r in self.range]

    @property
    def range(self):
        """list of 3-tuple: ranges for slicing given by (min, max, step)."""
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
            self.data, self._edge_width, self.length
        )
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles

        self.events.edge_width()
        self.refresh()

    @property
    def length(self) -> Union[int, float]:
        return self._length

    @length.setter
    def length(self, length: Union[int, float]):
        """float: Multiplicative factor for length of all vectors."""
        self._length = length

        vertices, triangles = generate_vector_meshes(
            self.data, self.edge_width, self._length
        )
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles

        self.events.length()
        self.refresh()

    @property
    def edge_color(self) -> str:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: str):
        """str: edge color of all the vectors."""
        self._edge_color = edge_color
        self.refresh()

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""

        vertices = self._mesh_vertices

        if len(self.data) == 0:
            faces = []
            self._data_view = np.empty((0, 2, 2))
        elif self.ndim > 2:
            matches = np.all(self.indices[:-2] == self.data[:, 0, :-2], axis=1)
            matches = np.where(matches)[0]
            self._data_view = self.data[matches, :, -2:]
            if len(matches) == 0:
                faces = []
            else:
                keep_inds = np.repeat(2 * matches, 2)
                keep_inds[1::2] = keep_inds[1::2] + 1
                faces = self._mesh_triangles[keep_inds]
        else:
            faces = self._mesh_triangles
            self._data_view = self.data[:, :, -2:]

        if len(faces) == 0:
            self._node.set_data(vertices=None, faces=None)
        else:
            self._node.set_data(
                vertices=vertices[:, ::-1] + 0.5,
                faces=faces,
                color=self.edge_color,
            )

        self._need_visual_update = True
        self._update()
        self._update_thumbnail()

    def _update_thumbnail(self):
        """Update thumbnail with current points and colors."""
        # calculate min vals for the vertices and pad with 0.5
        # the offset is needed to ensure that the top left corner of the
        # vectors corresponds to the top left corner of the thumbnail
        offset = np.array([self.range[-2][0], self.range[-1][0]]) + 0.5
        # calculate range of values for the vertices and pad with 1
        # padding ensures the entire vector can be represented in the thumbnail
        # without getting clipped
        shape = np.ceil(
            [
                self.range[-2][1] - self.range[-2][0] + 1,
                self.range[-1][1] - self.range[-1][0] + 1,
            ]
        ).astype(int)
        zoom_factor = np.divide(self._thumbnail_shape[:2], shape).min()

        vectors = copy(self._data_view)
        vectors[:, 1, :] = vectors[:, 0, :] + vectors[:, 1, :] * self.length
        downsampled = (vectors - offset) * zoom_factor
        downsampled = np.clip(
            downsampled, 0, np.subtract(self._thumbnail_shape[:2], 1)
        )
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1
        col = Color(self.edge_color).rgba
        if len(downsampled) > self._max_vectors_thumbnail:
            inds = np.random.randint(
                0, len(downsampled), self._max_vectors_thumbnail
            )
            downsampled = downsampled[inds]
        for v in downsampled:
            start = v[0]
            stop = v[1]
            step = int(np.ceil(np.max(abs(stop - start))))
            x_vals = np.linspace(start[0], stop[0], step)
            y_vals = np.linspace(start[1], stop[1], step)
            for x, y in zip(x_vals, y_vals):
                colormapped[int(x), int(y), :] = col
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
        edge_color = (255 * Color(self.edge_color).rgba).astype(np.int)
        stroke = f'rgb{tuple(edge_color[:3])}'
        opacity = str(self.opacity)
        props = {'stroke': stroke, 'stroke-width': width, 'opacity': opacity}

        for v in self._data_view:
            x1 = str(v[0, -2])
            y1 = str(v[0, -1])
            x2 = str(v[0, -2] + self.length * v[1, -2])
            y2 = str(v[0, -1] + self.length * v[1, -1])

            element = Element('line', x1=y1, y1=x1, x2=y2, y2=x2, **props)
            xml_list.append(element)

        return xml_list
