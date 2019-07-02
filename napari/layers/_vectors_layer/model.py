from typing import Union
from xml.etree.ElementTree import Element

import numpy as np
from copy import copy

from .._base_layer import Layer
from ..._vispy.scene.visuals import Mesh
from ...util.event import Event
from ...util import segment_normal
from .vectors_util import vectors_to_coordinates
from vispy.color import get_color_names, Color


class Vectors(Layer):
    """
    Vectors layer renders lines onto the image.

    Parameters
    ----------
    vectors : (N, 2, D) or (N1, N2, ..., ND, D) array
        An (N, 2, D) array is interpreted as "coordinate-like" data and a
        list of N vectors with start point and projections of the vector in
        D dimensions. An (N1, N2, ..., ND, D) array is interpreted as
        "image-like" data where there is a length D vector of the
        projections at each pixel.
    width : int
        width of the line in pixels
    length : float
        multiplier on length of the line
    color : str
        one of "get_color_names" from vispy.color
    mode : str
        control panel mode
    """

    # The max number of vectors that will ever be used to render the thumbnail
    # If more vectors are present then they are randomly subsampled
    _max_vectors_thumbnail = 1024

    def __init__(self, vectors, width=1, color='red', length=1, name=None):

        super().__init__(Mesh(), name)

        # events for non-napari calculations
        self.events.add(length=Event, width=Event)

        # Save the line style params
        self._width = width
        self._color = color
        self._colors = get_color_names()

        # Data containing vectors in the currently viewed slice
        self._vectors_view = np.empty((0, 2, 2))

        # length attribute
        self._length = length

        # update flags
        self._need_display_update = False
        self._need_visual_update = False

        # assign vector data and establish default behavior
        with self.freeze_refresh():
            self.data = vectors

        self.events.opacity.connect(lambda e: self._update_thumbnail())

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, vectors: np.ndarray):
        """(N, 2, D) or (N1, N2, ..., ND, D) array: a (N, 2, D) array is
        interpreted as "coordinate-like" data and a list of N vectors with start
        point and projections of the vector in D dimensions. A
        (N1, N2, ..., ND, D) array is interpreted as "image-like" data where
        there is a length D vector of the projections at each pixel.

        Parameters
        ----------
        vectors : (N, 2, D) array
        """

        self._data = vectors_to_coordinates(vectors)

        vertices, triangles = self._generate_meshes(
            self._data, self.width, self.length
        )
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles

        self.events.data()
        self.refresh()

    def _get_shape(self):
        return [r[1] for r in self.range]

    @property
    def range(self):
        """list of 3-tuple of int: ranges of data for slicing specifed by
        (min, max, step).
        """
        if len(self.data) == 0:
            maxs = np.ones(self.data.ndim, dtype=int)
            mins = np.zeros(self.data.ndim, dtype=int)
        else:
            # Convert from projections to endpoints using the current length
            data = copy(self.data)
            data[:, 1, :] = data[:, 0, :] + self.length * data[:, 1, :]
            maxs = np.max(data, axis=(0, 1))
            mins = np.min(data, axis=(0, 1))

        return [(min, max, 1) for min, max in zip(mins, maxs)]

    @property
    def width(self) -> Union[int, float]:
        return self._width

    @width.setter
    def width(self, width: Union[int, float]):
        """width of the line in pixels
        widths greater than 1px only guaranteed to work with "agg" method
        """
        self._width = width

        vertices, triangles = self._generate_meshes(
            self.data, self._width, self.length
        )
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles

        self.events.width()
        self.refresh()

    @property
    def length(self) -> Union[int, float]:
        return self._length

    @length.setter
    def length(self, length: Union[int, float]):
        """Change the length of all lines

        Parameters
        ----------
        length : int or float multiplicative factor
        """
        self._length = length

        vertices, triangles = self._generate_meshes(
            self.data, self.width, self._length
        )
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles

        self.events.length()
        self.refresh()

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, color: str):
        """Color, ColorArray: color of the body of the marker
        """
        self._color = color
        self.refresh()

    @property
    def svg_props(self):
        """dict: color and width properties in the svg specification
        """
        width = str(self.width)
        edge_color = (255 * Color(self.color).rgba).astype(np.int)
        stroke = f'rgb{tuple(edge_color[:3])}'
        opacity = str(self.opacity)

        props = {'stroke': stroke, 'stroke-width': width, 'opacity': opacity}

        return props

    def _generate_meshes(self, vectors, width, length):
        """Generates list of mesh vertices and triangles from a list of vectors

        Parameters
        ----------
        vectors : (N, 2, D) array
            A list of N vectors with start point and projections of the vector
            in D dimensions. Vectors are projected onto the last two
            dimensions if D > 2.
        width : float
            width of the line to be drawn
        length : float
            length multiplier of the line to be drawn

        Returns
        ----------
        vertices : (4N, 2) array
            Vertices of all triangles for the lines
        triangles : (2N, 2) array
            Vertex indices that form the mesh triangles
        """
        vectors = np.reshape(copy(vectors[:, :, -2:]), (-1, 2))
        vectors[1::2] = vectors[::2] + length * vectors[1::2]
        centers = np.repeat(vectors, 2, axis=0)
        offsets = segment_normal(vectors[::2, :], vectors[1::2, :])
        offsets = np.repeat(offsets, 4, axis=0)
        signs = np.ones((len(offsets), 2))
        signs[::2] = -1
        offsets = offsets * signs

        vertices = centers + width * offsets / 2
        triangles = np.array(
            [
                [2 * i, 2 * i + 1, 2 * i + 2]
                if i % 2 == 0
                else [2 * i - 1, 2 * i, 2 * i + 1]
                for i in range(len(vectors))
            ]
        ).astype(np.uint32)

        return vertices, triangles

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""

        vertices = self._mesh_vertices

        if len(self.data) == 0:
            faces = []
            self._vectors_view = np.empty((0, 2, 2))
        elif self.ndim > 2:
            matches = np.all(self.indices[:-2] == self.data[:, 0, :-2], axis=1)
            matches = np.where(matches)[0]
            self._vectors_view = self.data[matches, :, -2:]
            if len(matches) == 0:
                faces = []
            else:
                keep_inds = np.repeat(2 * matches, 2)
                keep_inds[1::2] = keep_inds[1::2] + 1
                faces = self._mesh_triangles[keep_inds]
        else:
            faces = self._mesh_triangles
            self._vectors_view = self.data[:, :, -2:]

        if len(faces) == 0:
            self._node.set_data(vertices=None, faces=None)
        else:
            self._node.set_data(
                vertices=vertices[:, ::-1] + 0.5, faces=faces, color=self.color
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

        vectors = copy(self._vectors_view)
        vectors[:, 1, :] = vectors[:, 0, :] + vectors[:, 1, :] * self.length
        downsampled = (vectors - offset) * zoom_factor
        downsampled = np.clip(
            downsampled, 0, np.subtract(self._thumbnail_shape[:2], 1)
        )
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1
        col = Color(self.color).rgba
        if len(downsampled) > self._max_vectors_thumbnail:
            inds = np.random.randint(
                0, len(downsampled), self._max_vectors_thumbnail
            )
            downsampled = downsampled[inds]
        for v in downsampled:
            start = v[0]
            stop = v[1]
            step = np.ceil(np.max(abs(stop - start)))
            x_vals = np.linspace(start[0], stop[0], step)
            y_vals = np.linspace(start[1], stop[1], step)
            for x, y in zip(x_vals, y_vals):
                colormapped[int(x), int(y), :] = col
        colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    def to_xml_list(self):
        """Convert the vectors to a list of xml elements according to the svg
        specification. Each vector is represented by a line.

        Returns
        ----------
        xml : list
            List of xml elements defining each marker according to the
            svg specification
        """
        xml_list = []

        for v in self._vectors_view:
            x1 = str(v[0, -2])
            y1 = str(v[0, -1])
            x2 = str(v[0, -2] + self.length * v[1, -2])
            y2 = str(v[0, -1] + self.length * v[1, -1])

            element = Element(
                'line', x1=y1, y1=x1, x2=y2, y2=x2, **self.svg_props
            )
            xml_list.append(element)

        return xml_list
