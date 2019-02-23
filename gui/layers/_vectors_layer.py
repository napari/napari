#!/usr/bin/env python
# title           : _vectors_layer.py
# description     :class Vectors layer that defines properties
# author          :bryant.chhun
# date            :1/16/19
# version         :0.0
# usage           :
# notes           :
# python_version  :3.6

from typing import Union

import numpy as np

from ._base_layer import Layer
from ._register import add_to_viewer
from .._vispy.scene.visuals import Line as LinesNode
from vispy.color import get_color_names

from .qt import QtVectorsLayer

@add_to_viewer
class Vectors(Layer):
    """
    Properties
    ----------
    vectors : np.ndarray
        array of shape (N,2) or (N,3) = array of 2d or 3d coordinates

    data : np.ndarray
        ABC implementation.  Same as vectors

    averaging : str
        kernel over which to average from one of _avg_dims
        averaging.setter adjusts the underlying data and must be implemented per use case
        subscribe an observer by registering it with "averaging_bind_to"

    width : int
        width of the line in pixels

    length : int or float
        length of the line
        length.setter adjusts the underlying data and must be implemented per use case
        subscribe an observer by registering it with "length_bind_to"

    color : str
        one of "get_color_names" from vispy.color

    conector : str or np.ndarray
        way to render line between coordinates
        two built in types: "segment" or "strip"
        third type is np.ndarray

    method : str
        Render method: one of 'agg' or 'gl'
        used by vispy.LineVisual

    antialias : bool
        Antialias rendering
        used by vispy.LineVisual

    Attributes
    ----------
        Private (not associated with property)
        -------
        _connector_types
        _colors
        _avg_dims
        _avg_observers
        _len_observers
        _need_display_update
        _need_visual_update
        
        Public
        -------
        name


    See vispy's line visual docs for more details:
    http://api.vispy.org/en/latest/visuals.html#vispy.visuals.LineVisual
    http://vispy.org/visuals.html
    """

    def __init__(
        self, vectors,
            width=1,
            color='red',
            connector='segments',
            averaging='1x1',
            length = 10):
            # arrow_size=10):

        visual = LinesNode()
        # visual = ArrowNode()
        # visual = ArrowNode(arrow_size=10, color='white')
        super().__init__(visual)

        # Save the line vertex coordinates
        # self._vectors = vectors
        self._vectors = self.check_vector_type(vectors)
        # self._arrows = None
        # self._arrow_size = arrow_size

        # Save the line style params
        self._width = width
        self._color = color
        self._connector = connector

        self._connector_types = ['segments']
        self._colors = get_color_names()
        
        self._avg_dims = ['1x1','3x3','5x5','7x7','9x9','11x11']
        self._averaging = averaging
        self._length = length
        self._avg_observers = []
        self._len_observers = []

        # update flags
        self._need_display_update = False
        self._need_visual_update = False

        self.name = 'vectors'
        self._qt = QtVectorsLayer(self)

    #====================== Property getter and setters =======================================
    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    @vectors.setter
    def vectors(self, vectors: np.ndarray):
        """
        Can accept two data types:
            1) (N, 4) array with elements (x, y, u, v),
                where x-y are position (center) and u-v are x-y projections of the vector
            2) (N, M, 2) array with elements (u, v)
                where u-v are x-y projections of the vector
                vector position is one per-pixel in the NxM array
        :param vectors: ndarray
        :return:
        """
        self._vectors = self.check_vector_type(vectors)

        self.viewer._child_layer_changed = True
        self._refresh()

    def check_vector_type(self, vectors):
        if vectors.shape[-1] == 4 and len(vectors.shape) == 2:
            coord_list = self._convert_proj_to_coordinates(vectors)
        elif vectors.shape[-1] == 2 and len(vectors.shape) == 3:
            coord_list = self._convert_matrix_to_coordinates(vectors)
            # raise NotImplementedError("image-like vector data is not supported")
        else:
            raise NotImplementedError("Vector data of shape %s is not supported" % str(vectors.shape))
        return coord_list

    def _convert_matrix_to_coordinates(self, vect) -> np.ndarray:
        """
        To convert an image-like array of (x-proj, y-proj) into an
            image-like array of vectors

        :param vect: np.ndarray of shape (N, M, 2)
        :return: position list of shape (2*N*M, 2) for vispy
        """
        xdim = vect.shape[0]
        ydim = vect.shape[1]

        # stride is used during averaging for a later implementation
        # stride_x = 2
        # stride_y = 2

        # create empty vector of necessary shape
        # every pixel has 2 coordinates,
        pos = np.empty((2 * xdim * ydim, 2), dtype=np.float32)

        # create coordinate spacing for x-y
        # double the num of elements by doubling x sampling
        xspace = np.linspace(0, xdim, 2 * xdim)
        yspace = np.linspace(0, ydim, ydim)
        xv, yv = np.meshgrid(xspace, yspace)

        # assign coordinates (pos) to all pixels
        pos[:, 0] = xv.flatten()
        pos[:, 1] = yv.flatten()

        # adjust second coordinate to represent vector projections
        pos[1::2, 0] += vect.reshape((xdim*ydim, 2))[:, 0]
        pos[1::2, 1] += vect.reshape((xdim*ydim, 2))[:, 1]

        # # TODO: averaging implementation.  for image-like arrays only
        # pos[1::2, 0] += (stride_x / 2) * vect.reshape((xdim*ydim, 2))[:, 0]
        # pos[1::2, 1] += (stride_y / 2) * vect.reshape((xdim*ydim, 2))[:, 1]

        return pos

    def _convert_proj_to_coordinates(self, vect) -> np.ndarray:
        """
        To convert a list of coordinates of shape (x-center, y-center, x-proj, y-proj)
            into a position list of vectors.
        Every input coordinate of (N,4) results in two output coordinates of (N,2)

        :param vect: np.ndarray of shape (N, 4)
        :return: position list of shape (2*N, 2) for vispy
        """

        # create empty vector of necessary shape
        #   one coordinate for each endpoint of the vector
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
    def arrow_size(self):
        return self._arrow_size

    @arrow_size.setter
    def arrow_size(self, val: int):
        self._arrow_size = val
        self._need_visual_update = True
        self._refresh()

    @property
    def arrows(self):
        return self._arrows

    @arrows.setter
    def arrows(self, arrow_pos):
        self._arrows = arrow_pos
        self._need_visual_update = True
        self._refresh()

    @property
    def averaging(self) -> str:
        """
        Set the kernel over which to average
        :return: string of averaging kernel size
        """
        return self._averaging
    
    @averaging.setter
    def averaging(self, averaging: str):
        '''
        Calculates an average vector over a kernel
        Averaging does nothing unless the user binds an observer and updates
            the underlying data manually.
        :param averaging: one of "_avg_dims" above
        :return: None
        '''
        self._averaging = averaging
        for callback in self._avg_observers:
            print('averaging changed, broadcasting to subscribers')
            callback(self._averaging)

    def averaging_bind_to(self, callback):
        self._avg_observers.append(callback)
        
    @property
    def width(self) -> Union[int, float]:
        """
        width of the line in pixels
            widths greater than 1px only guaranteed to work with "agg" method
        :return: int or float line width
        """
        return self._width

    @width.setter
    def width(self, width: Union[int, float]) -> None:
        self._width = width
        self._refresh()

    @property
    def length(self) -> Union[int, float]:
        return self._length

    @length.setter
    def length(self, length: Union[int, float]):
        """
        Length of the line.
        Does nothing unless the user binds an observer and updates
            the underlying data manually
        :param magnitude: length multiplicative factor
        :return: None
        """

        print('length setter called, new length = '+str(length))
        self._length = length
        for callback in self._len_observers:
            print('length changed, broadcasting to subscribers')
            callback(self._length)
        # self._refresh()

    def length_bind_to(self, callback):
        self._len_observers.append(callback)

    @property
    def color(self) -> str:
        """Color, ColorArray: color of the body of the marker
        """
        return self._color

    @color.setter
    def color(self, color: str) -> None:
        self._color = color
        self._refresh()

    @property
    def connector(self):
        """
        line connector.  One of _connector.types = "strip", "segments"
        or can receive an ndarray.
        :return: connector parameter
        """
        return self._connector

    @connector.setter
    def connector(self, connector_type: Union[str, np.ndarray]):
        self._connector = connector_type
        self._refresh()

    # =========================== Napari Layer ABC methods =====================
    @property
    def data(self) -> np.ndarray:
        """

        :return: coordinates of line vertices via the property (and not the private)
        """
        return self.vectors

    @data.setter
    def data(self, data: np.ndarray) -> None:
        """
        Set the data via the property, which calls reformatters (and not by altering the private)
        :param data:
        :return:
        """
        self.vectors = data


    def _get_shape(self):
        if len(self.vectors) == 0:
            return np.ones(self.vectors.shape,dtype=int)
        else:
            return np.max(self.vectors, axis=0) + 1

    def _refresh(self):
        """Fully refresh the underlying visual.
        """
        self._need_display_update = True
        self._update()

    def _update(self):
        """Update the underlying visual.
        """
        if self._need_display_update:
            self._need_display_update = False

            self._set_view_slice(self.viewer.dimensions.indices)

        if self._need_visual_update:
            self._need_visual_update = False
            self._node.update()


    def _set_view_slice(self, indices):
        """Sets the view given the indices to slice with.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.
        """
        in_slice_vectors, matches = self._slice_vectors(indices)

        # Display vectors if there are any in this slice
        if len(in_slice_vectors) > 0:
            # Get the vectors sizes
            # print('in_slice_vectors greater than zero')
            # if isinstance(self.width, (list, np.ndarray)):
            #     sizes = self.width[matches][::-1]
            #
            # else:
            #     sizes = self.width

            # Update the vectors node
            data = np.array(in_slice_vectors) + 0.5

        else:
            # if no vectors in this slice send dummy data
            data = np.empty((0, 2))
            sizes = 0

        self._node.set_data(
            data[::-1],
            width=self.width,
            color=self.color,
            connect=self.connector)
        self._need_visual_update = True
        self._update()

    def _slice_vectors(self, indices):
        """Determines the slice of markers given the indices.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.
        """
        # Get a list of the vectors for the vectors in this slice

        vectors = self.vectors
        if len(vectors) > 0:
            matches = np.equal(
                vectors[:, 2:],
                np.broadcast_to(indices[2:], (len(vectors),len(indices) - 2) )
            )

            matches = np.all(matches, axis=1)

            in_slice_vectors = vectors[matches, :2]
            return in_slice_vectors, matches
        else:
            return [], []



class InvalidDataFormatError(Exception):
    """
    To better describe when vector data is not correct
        more informative than TypeError
    """

    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message