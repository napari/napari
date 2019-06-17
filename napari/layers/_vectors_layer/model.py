from typing import Union
from xml.etree.ElementTree import Element

import numpy as np
from scipy import signal

from .._base_layer import Layer
from ..._vispy.scene.visuals import Mesh
from ...util.event import Event
from ...util import segment_normal
from vispy.color import get_color_names, Color


class Vectors(Layer):
    """
    Vectors layer renders lines onto the image.

    Parameters
    ----------
    vectors : (N, 2, D) or (N1, N2, ..., ND, D) array
        A (N, 2, D) array is interpted as "coordinate-like" data and a list of
        N vectors with start point and projections of the vector in D
        dimensions. A (N1, N2, ..., ND, D) array is interpted as
        "image-like" data where there is a length D vector of the projections
        at each pixel.
    averaging : int
        Size of kernel over which to convolve and subsample the data not
        implemented for "coordinate-like" data
    width : int
        width of the line in pixels
    length : float
        multiplier on length of the line
    color : str
        one of "get_color_names" from vispy.color
    mode : str
        control panel mode
    """

    def __init__(
        self, vectors, width=1, color='red', averaging=1, length=1, name=None
    ):

        visual = Mesh()
        super().__init__(visual)

        # events for non-napari calculations
        self.events.add(length=Event, width=Event, averaging=Event)

        # Store underlying data model
        self._data_types = ('image', 'coords')
        self._data_type = None

        # Save the line style params
        self._width = width
        self._color = color
        self._colors = get_color_names()

        # averaging and length attributes
        self._averaging = averaging
        self._length = length

        # update flags
        self._need_display_update = False
        self._need_visual_update = False

        # assign vector data and establish default behavior
        self._raw_data = vectors
        with self.freeze_refresh():
            self.vectors = vectors

    # ====================== Property getter and setters =====================

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    @vectors.setter
    def vectors(self, vectors: np.ndarray):
        """(N, 2, D) or (N1, N2, ..., ND, D) array: a (N, 2, D) array is
        interpted as "coordinate-like" data and a list of N vectors with start
        point and projections of the vector in D dimensions. A
        (N1, N2, ..., ND, D) array is interpted as "image-like" data where
        there is a length D vector of the projections at each pixel.

        Parameters
        ----------
        vectors : (N, 2, D) array
        """

        self._vectors = self._convert_to_vector_type(vectors)

        vertices, triangles = self._generate_meshes(
            self._vectors, self.width, self.length
        )
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles

        self.events.data()
        self.refresh()

    def _convert_to_vector_type(self, vectors):
        """Check on input data for proper shape and dtype

        Parameters
        ----------
        vectors : (N, 2, D) or (N1, N2, ..., ND, D) array
            A (N, 2, D) array is interpted as "coordinate-like" data and a list
            of N vectors with start point and projections of the vector in D
            dimensions. A (N1, N2, ..., ND, D) array is interpted as
            "image-like" data where there is a length D vector of the
            projections at each pixel.

        Returns
        coords : (N, 2, D) array
            A list of N vectors with start point and projections of the vector
            in D dimensions.
        """
        if vectors.shape[-2] == 2 and vectors.ndim == 3:
            # an (N, 2, D) array that is coordinate-like
            coords = vectors
            self._data_type = self._data_types[1]

        elif vectors.shape[-1] == vectors.ndim - 1:
            # an (N1, N2, ..., ND, D) array that is image-like
            coords = self._convert_image_to_coordinates(vectors)
            self._data_type = self._data_types[0]

        else:
            raise TypeError(
                "Vector data of shape %s is not supported" % str(vectors.shape)
            )

        return coords

    def _convert_image_to_coordinates(self, vect):
        """To convert an image-like array with elements (y-proj, x-proj) into a
        position list of coordinates
        Every pixel position (n, m) results in two output coordinates of (N,2)

        Parameters
        ----------
        vectors : (N1, N2, ..., ND, D) array
            "image-like" data where there is a length D vector of the
            projections at each pixel.

        Returns
        ----------
        coords : (N, 2, D) array
            A list of N vectors with start point and projections of the vector
            in D dimensions.
        """
        xdim = vect.shape[0]
        ydim = vect.shape[1]

        # stride is used during averaging and length adjustment
        stride_x, stride_y = self._averaging, self._averaging

        # create empty vector of necessary shape
        # every "pixel" has 2 coordinates
        pos = np.empty((2 * xdim * ydim, 2), dtype=np.float32)

        # create coordinate spacing for x-y
        # double the num of elements by doubling x sampling
        xspace = np.linspace(0, stride_x * xdim, 2 * xdim, endpoint=False)
        yspace = np.linspace(0, stride_y * ydim, ydim, endpoint=False)
        xv, yv = np.meshgrid(xspace, yspace)

        # assign coordinates (pos) to all pixels
        pos[:, 0] = xv.flatten()
        pos[:, 1] = yv.flatten()

        # pixel midpoints are the first x-values of positions
        midpt = np.zeros((xdim * ydim, 2), dtype=np.float32)
        midpt[:, 0] = pos[0::2, 0] + (stride_x - 1) / 2
        midpt[:, 1] = pos[0::2, 1] + (stride_y - 1) / 2

        # rotate coordinates about midpoint to represent angle and length
        pos[0::2, 0] = (
            midpt[:, 0]
            - (stride_x / 2)
            * (self._length / 2)
            * vect.reshape((xdim * ydim, 2))[:, 0]
        )
        pos[0::2, 1] = (
            midpt[:, 1]
            - (stride_y / 2)
            * (self._length / 2)
            * vect.reshape((xdim * ydim, 2))[:, 1]
        )
        pos[1::2, 0] = (
            midpt[:, 0]
            + (stride_x / 2)
            * (self._length / 2)
            * vect.reshape((xdim * ydim, 2))[:, 0]
        )
        pos[1::2, 1] = (
            midpt[:, 1]
            + (stride_y / 2)
            * (self._length / 2)
            * vect.reshape((xdim * ydim, 2))[:, 1]
        )

        return pos

    @property
    def averaging(self) -> int:
        return self._averaging

    @averaging.setter
    def averaging(self, value: int):
        """Calculates an average vector over a kernel

        Parameters
        ----------
        value : int that defines (int, int) kernel
        """
        self._averaging = value

        self.events.averaging()
        self._update_avg()

        self.refresh()

    def _update_avg(self):
        """Method for calculating average
        Implemented ONLY for image-like vector data
        """
        if self._data_type == 'coords':
            # default averaging is supported only for 'matrix' dataTypes
            return
        elif self._data_type == 'image':

            x, y = self._averaging, self._averaging

            if (x, y) == (1, 1):
                self.vectors = self._original_data
                # calling original data
                return

            tempdat = self._original_data
            range_x = tempdat.shape[0]
            range_y = tempdat.shape[1]
            x_offset = int((x - 1) / 2)
            y_offset = int((y - 1) / 2)

            kernel = np.ones(shape=(x, y)) / (x * y)

            output_mat = np.zeros_like(tempdat)
            output_mat_x = signal.convolve2d(
                tempdat[:, :, 0], kernel, mode='same', boundary='wrap'
            )
            output_mat_y = signal.convolve2d(
                tempdat[:, :, 1], kernel, mode='same', boundary='wrap'
            )

            output_mat[:, :, 0] = output_mat_x
            output_mat[:, :, 1] = output_mat_y

            self.vectors = output_mat[
                x_offset : range_x - x_offset : x,
                y_offset : range_y - y_offset : y,
            ]

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
            self.vectors, self._width, self.length
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
            self.vectors, self.width, self._length
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

    # =========================== Napari Layer ABC methods ===================
    @property
    def data(self) -> np.ndarray:
        return self.vectors

    @data.setter
    def data(self, data: np.ndarray):
        self.vectors = data

    def _get_shape(self):
        if len(self.vectors) == 0:
            return np.ones(2, dtype=int)
        else:
            return np.max(self.vectors, axis=0) + 1

    @property
    def range(self):
        """list of 3-tuple of int: ranges of data for slicing specifed by
        (min, max, step).
        """
        if len(self.vectors) == 0:
            maxs = [1, 1]
            mins = [0, 0]
        else:
            maxs = np.max(self.vectors[:, 0, :], axis=0) + 1
            mins = np.min(self.vectors[:, 0, :], axis=0)

        return [(min, max, 1) for min, max in zip(mins, maxs)]

    def _generate_meshes(self, vectors, width, length):
        """Generates list of mesh vertices and triangles from a list of vectors

        Parameters
        ----------
        vectors : (N, 2, D) array
            A list of N vectors with start point and projections of the vector
            in D dimensions.
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
        vectors = np.reshape(vectors, (-1, vectors.shape[-1]))
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
        faces = self._mesh_triangles

        if len(faces) == 0:
            self._node.set_data(vertices=None, faces=None)
        else:
            self._node.set_data(
                vertices=vertices[:, ::-1], faces=faces, color=self.color
            )

        self._need_visual_update = True
        self._update()

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

        for v in self.vectors:
            x1 = str(v[0, 0])
            y1 = str(v[0, 1])
            x2 = str(v[0, 0] + self.length * v[1, 0])
            y2 = str(v[0, 1] + self.length * v[1, 1])

            element = Element(
                'line', x1=y1, y1=x1, x2=y2, y2=x2, **self.svg_props
            )
            xml_list.append(element)

        return xml_list
