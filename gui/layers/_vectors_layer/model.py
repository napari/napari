#!/usr/bin/env python
# title           : properties.py
# description     :class Vectors layer that defines properties
# author          :bryant.chhun
# date            :1/16/19
# version         :0.0
# usage           :
# notes           :
# python_version  :3.6

from typing import Union

import numpy as np

from .._base_layer import Layer
from .._register import add_to_viewer
from gui._vispy.scene.visuals import Line as LinesNode
from vispy.color import get_color_names

from vispy.util.event import Event

from .view import QtVectorsLayer
from .view import QtVectorsControls

import cv2

@add_to_viewer
class Vectors(Layer):
    """
    Properties
    ----------
    vectors : np.ndarray
        array of shape (N,4) or (N, M , 2)

    data : np.ndarray
        ABC implementation.  Same as vectors

    averaging : str
        kernel over which to average from one of _kernel_dict
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
        Private
        -------

        _connector_types
        _colors
        _kernel_dict
        _kernel
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
            length=1):

        visual = LinesNode()
        super().__init__(visual)

        # map averaging type to tuple
        self._kernel_dict = {'1x1': (1, 1),
                             '3x3': (3, 3),
                             '5x5': (5, 5),
                             '7x7': (7, 7),
                             '9x9': (9, 9),
                             '11x11': (11, 11)}
        self._kernel = self._kernel_dict['1x1']

        # Store underlying data model
        self._data_types = ('matrix', 'projection')
        self._data_type = None

        # Save the line style params
        self._width = width
        self._color = color
        self._connector_types = ['segments']
        self._connector = connector
        self._colors = get_color_names()

        # averaging and length attributes
        self._avg_dims = ['1x1', '3x3', '5x5', '7x7', '9x9', '11x11']
        self._averaging = averaging
        self._length = length
        self._avg_observers = []
        self._len_observers = []

        # update flags
        self._need_display_update = False
        self._need_visual_update = False

        # assign vector data and establish default behavior
        self._raw_dat = None
        self._original_data = vectors
        self._vectors = self._check_vector_type(vectors)
        self.averaging_bind_to(self._default_avg)
        self.length_bind_to(self._default_length)

        self._mode = 'pan/zoom'
        self._mode_history = self._mode

        self.name = 'vectors'
        # self._qt = QtVectorsLayer(self)
        self.events.add(mode=Event)
        self._qt_properties = QtVectorsLayer(self)
        self._qt_controls = QtVectorsControls(self)


    #====================== Property getter and setters =======================================
    @property
    def _original_data(self):
        return self._raw_dat

    @_original_data.setter
    def _original_data(self, dat):
        if self._raw_dat is None:
            self._raw_dat = dat

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
        print('vector setter being called')
        self._original_data = vectors

        self._vectors = self._check_vector_type(vectors)

        self.viewer._child_layer_changed = True
        self._refresh()

    def _check_vector_type(self, vectors):
        """
        check on input data for proper shape and dtype
        :param vectors: ndarray
        :return:
        """
        if vectors.shape[-1] == 4 and len(vectors.shape) == 2:
            coord_list = self._convert_proj_to_coordinates(vectors)
            self._data_type = self._data_types[1]
        elif vectors.shape[-1] == 2 and len(vectors.shape) == 3:
            coord_list = self._convert_matrix_to_coordinates(vectors)
            self._data_type = self._data_types[0]
        else:
            raise InvalidDataFormatError("Vector data of shape %s is not supported" % str(vectors.shape))

        if vectors.shape[-1] == 4:
            print("four")
            #check that this is range -1 to 1
        elif vectors.shape[-1] == 2:
            print("two")
            #check that this is in range -1 to 1
        return coord_list

    def _convert_matrix_to_coordinates(self, vect) -> np.ndarray:
        """
        To convert an image-like array of (x-proj, y-proj) into an
            image-like array of vectors

        :param vect: ndarray of shape (N, M, 2)
        :return: position list of shape (2*N*M, 2) for vispy
        """
        xdim = vect.shape[0]
        ydim = vect.shape[1]

        # stride is used during averaging and length adjustment
        stride_x = self._kernel[0]
        stride_y = self._kernel[1]
        print('calling matrix to coords using kernel/length'+str(stride_x)+str(stride_y)+str(self._length))

        # create empty vector of necessary shape
        # every "pixel" has 2 coordinates
        pos = np.empty((2 * xdim * ydim, 2), dtype=np.float32)

        # create coordinate spacing for x-y
        # double the num of elements by doubling x sampling
        xspace = np.linspace(0, stride_x*xdim, 2 * xdim)
        yspace = np.linspace(0, stride_y*ydim, ydim)
        xv, yv = np.meshgrid(xspace, yspace)

        # assign coordinates (pos) to all pixels
        pos[:, 0] = xv.flatten()
        pos[:, 1] = yv.flatten()

        # pixel midpoints are the first x-values of positions
        midpt = np.zeros((xdim * ydim, 2), dtype=np.float32)
        midpt[:, 0] = pos[0::2, 0]
        midpt[:, 1] = pos[0::2, 1]

        # rotate coordinates about midpoint to represent angle and length
        pos[0::2, 0] = midpt[:, 0] - (stride_x / 2) * (self._length/2) * vect.reshape((xdim*ydim, 2))[:, 0]
        pos[0::2, 1] = midpt[:, 1] - (stride_y / 2) * (self._length/2) * vect.reshape((xdim*ydim, 2))[:, 1]
        pos[1::2, 0] = midpt[:, 0] + (stride_x / 2) * (self._length/2) * vect.reshape((xdim*ydim, 2))[:, 0]
        pos[1::2, 1] = midpt[:, 1] + (stride_y / 2) * (self._length/2) * vect.reshape((xdim*ydim, 2))[:, 1]

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
        :param averaging: one of "_avg_dims" above
        :return:
        '''
        self._averaging = averaging
        self._kernel = self._kernel_dict[averaging]

        print('averaging changed, broadcasting to subscribers')
        if self._default_avg in self._avg_observers:
            self._default_avg(averaging)
        else:
            for callback in self._avg_observers:
                callback(self._averaging)

        self._refresh()

    def averaging_bind_to(self, callback):
        '''
        register an observer to be notified upon changes to averaging
        Removes the default method for averaging if an external method is appended
        :param callback: target function
        :return:
        '''
        if self._default_avg in self._avg_observers:
            self._avg_observers.remove(self._default_avg)
        self._avg_observers.append(callback)

    def _default_avg(self, avg_kernel: str):
        '''
        Default method for calculating average
        :param avg_kernel: kernel over which to compute average
        :return:
        '''
        if self._data_type == 'projection':
            # default averaging is supported only for 'matrix' type data formats
            return None
        else:
            self._kernel = self._kernel_dict[avg_kernel]
            tempdat = self._original_data
            x = self._kernel[0]
            y = self._kernel[1]
            x_offset = int((x - 1) / 2)
            y_offset = int((y - 1) / 2)

            #if we allow a cv2 dependency:
            self._vectors = self._check_vector_type(cv2.blur(tempdat, (x, y))[x_offset:-x_offset - 1:x, y_offset:-y_offset - 1:y])

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
        print('length changed, broadcasting to subscribers')

        self._length = length

        if self._default_length in self._len_observers:
            self._default_length(length)
        else:
            for callback in self._avg_observers:
                callback(self._length)

        self._refresh()

    def length_bind_to(self, callback):
        '''

        :param callback:
        :return:
        '''
        if self._default_length in self._len_observers:
            self._len_observers.remove(self._default_length)
        else:
            self._len_observers.append(callback)

    def _default_length(self, newlen: int):
        '''

        :param newlen:
        :return:
        '''

        if self._data_type == 'projection':
            return None
        else:
            self._length = newlen
            self._vectors = self._check_vector_type(self._original_data)


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

    @property
    def mode(self):
        """None, str: Interactive mode
        """
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode == self.mode:
            return
        # if mode == 'add':
        #     self.cursor = 'cross'
        #     self.interactive = False
        #     self.help = 'hold <space> to pan/zoom'
        #     self.status = mode
        #     self._mode = mode
        # elif mode == 'select':
        #     self.cursor = 'pointing'
        #     self.interactive = False
        #     self.help = 'hold <space> to pan/zoom'
        #     self.status = mode
        #     self._mode = mode
        if mode == 'pan/zoom':
            self.cursor = 'standard'
            self.interactive = True
            self.help = ''
            self.status = mode
            self._mode = mode
        else:
            raise ValueError("Mode not recongnized")

        self.events.mode(mode=mode)

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

            self._set_view_slice(self.viewer.dims.indices)

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
            data = np.array(in_slice_vectors) + 0.5
        else:
            # if no vectors in this slice send dummy data
            data = np.empty((0, 2))

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

    # ========================= Napari Layer ABC CONTROL methods =====================

    def on_key_press(self, event):
        """Called whenever key pressed in canvas.
        """
        if event.native.isAutoRepeat():
            return
        else:
            if event.key == ' ':
                if self.mode != 'pan/zoom':
                    self._mode_history = self.mode
                    self.mode = 'pan/zoom'
                else:
                    self._mode_history = 'pan/zoom'
            elif event.key == 'Shift':
                if self.mode == 'add':
                    self.cursor = 'forbidden'
            elif event.key == 'a':
                self.mode = 'add'
            elif event.key == 's':
                self.mode = 'select'
            elif event.key == 'z':
                self.mode = 'pan/zoom'

    def on_key_release(self, event):
        """Called whenever key released in canvas.
        """
        if event.key == ' ':
            if self._mode_history != 'pan/zoom':
                self.mode = self._mode_history
        elif event.key == 'Shift':
            if self.mode == 'add':
                self.cursor = 'cross'


class InvalidDataFormatError(Exception):
    """
    To better describe when vector data is not correct
        more informative than TypeError
    """

    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message