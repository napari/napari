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
    vectors : np.ndarray of shape (N,4) or (N, M, 2)
        (N, 4) is a list of coordinates (y, x, v, u)
            x and y are coordinates
            u and v are y and x projections of the vector
        (N, M, 2) is an (N, M) image of (v, u) projections
        Returns np.ndarray of the current display (including averaging,
        length)
    averaging : int
        (int, int) kernel over which to convolve and subsample the data not
        implemented for (N, 4) data
    width : int
        width of the line in pixels
    length : float
        length of the line
        not implemented for (N, 4) data
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
        self._raw_data = None
        self._original_data = vectors
        self._current_data = vectors

        self._vectors = self._convert_to_vector_type(vectors)
        vertices, triangles = self._generate_meshes(self._vectors, self.width)
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles

        if name is None:
            self.name = 'vectors'
        else:
            self.name = name

    # ====================== Property getter and setters =====================
    @property
    def _original_data(self) -> np.ndarray:
        return self._raw_data

    @_original_data.setter
    def _original_data(self, data: np.ndarray):
        """Must preserve data used at construction. Specifically for default
        averaging/length adjustments.
        averaging/length adjustments recalculate the underlying data

        Parameters
        ----------
        data : np.ndarray
        """
        if self._raw_data is None:
            self._raw_data = data

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    @vectors.setter
    def vectors(self, vectors: np.ndarray):
        """Can accept two data types:
            1) (N, 4) array with elements (y, x, v, u),
                where x-y are position (center) and u-v are x-y projections of
                    the vector
            2) (N, M, 2) array with elements (v, u)
                where u-v are x-y projections of the vector
                vector position is one per-pixel in the NxM array

        Parameters
        ----------
        vectors : np.ndarray
        """
        self._original_data = vectors
        self._current_data = vectors

        self._vectors = self._convert_to_vector_type(self._current_data)

        vertices, triangles = self._generate_meshes(self._vectors, self.width)
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles

        self.events.data()
        self.refresh()

    def _convert_to_vector_type(self, vectors):
        """Check on input data for proper shape and dtype

        Parameters
        ----------
        vectors : np.ndarray
        """
        if vectors.shape[-1] == 4 and vectors.ndim == 2:
            coord_list = self._convert_coords_to_coordinates(vectors)
            self._data_type = self._data_types[1]

        elif vectors.shape[-1] == 2 and vectors.ndim == 3:
            coord_list = self._convert_image_to_coordinates(vectors)
            self._data_type = self._data_types[0]

        else:
            raise TypeError(
                "Vector data of shape %s is not supported" % str(vectors.shape)
            )

        return coord_list

    def _convert_image_to_coordinates(self, vect) -> np.ndarray:
        """To convert an image-like array with elements (y-proj, x-proj) into a
        position list of coordinates
        Every pixel position (n, m) results in two output coordinates of (N,2)

        Parameters
        ----------
        vect : np.ndarray of shape (N, M, 2)
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

    def _convert_coords_to_coordinates(self, vect) -> np.ndarray:
        """To convert a list of coordinates of shape
        (y-center, x-center, y-proj, x-proj) into a list of coordinates
        Input coordinate of (N,4) becomes two output coordinates of (N,2)

        Parameters
        ----------
        vect : np.ndarray of shape (N, 4)
        """
        # create empty vector of necessary shape
        # one coordinate for each endpoint of the vector
        pos = np.empty((2 * len(vect), 2), dtype=np.float32)

        # create pairs of points
        pos[0::2, 0] = vect[:, 0]
        pos[1::2, 0] = vect[:, 0]
        pos[0::2, 1] = vect[:, 1]
        pos[1::2, 1] = vect[:, 1]

        # adjust second of each pair according to x-y projection
        pos[1::2, 0] += vect[:, 2]
        pos[1::2, 1] += vect[:, 3]

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

        vertices, triangles = self._generate_meshes(self.vectors, self._width)
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
        self._update_length()
        self.events.length()

        self.refresh()

    def _update_length(self):
        """
        Method for calculating vector lengths
        Implemented ONLY for image-like vector data
        """

        if self._data_type == 'coords':
            return "length adjustment not allowed for coordinate-style data"
        elif self._data_type == 'image':
            self._vectors = self._convert_to_vector_type(self._current_data)
            vertices, triangles = self._generate_meshes(
                self.vectors, self.width
            )
            self._mesh_vertices = vertices
            self._mesh_triangles = triangles

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
            maxs = np.max(self.vectors, axis=0) + 1
            mins = np.min(self.vectors, axis=0)

        return [(min, max, 1) for min, max in zip(mins, maxs)]

    def _generate_meshes(self, vectors, width):
        """Generates list of mesh vertices and triangles from a list of vectors

        Parameters
        ----------
        vectors : np.ndarray
            Nx2 array where each pair of vertices corresponds to an independent
            line segment
        width : float
            width of the line to be drawn

        Returns
        ----------
        vertices : np.ndarray
            2Nx2 array of vertices of all triangles for the lines
        triangles : np.ndarray
            Nx3 array of vertex indices that form the mesh triangles
        """
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

        for i in range(len(self.vectors) // 2):
            x1 = str(self.vectors[2 * i, 0])
            y1 = str(self.vectors[2 * i, 1])
            x2 = str(self.vectors[2 * i + 1, 0])
            y2 = str(self.vectors[2 * i + 1, 1])

            element = Element(
                'line', x1=y1, y1=x1, x2=y2, y2=x2, **self.svg_props
            )
            xml_list.append(element)

        return xml_list
