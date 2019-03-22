
from typing import Union

import numpy as np
import scipy.signal as signal

from .._base_layer import Layer
from .._register import add_to_viewer
from ..._vispy.scene.visuals import Line as LinesNode
from vispy.color import get_color_names

from .view import QtVectorsLayer

@add_to_viewer
class Vectors(Layer):
    """
    Vectors layer renders lines onto the image.
    There are currently two data modes supported:
        1) image-like vectors where every pixel defines its own vector
        2) coordinate-like vectors where x-y position is not fixed to a grid
    Supports ONLY 2d vector field

    Properties
    ----------
    vectors : np.ndarray
        array of shape (N,4) or (N, M , 2)

    data : np.ndarray
        ABC implementation.
        returns vectors property setter/getter

    _original_data : np.ndarray
        Used by averaging and length adjustments.
        Is set during layer construction.
        Is NOT updated if data is adjusted by assigning the 'vectors' property

    averaging : str
        kernel over which to average from one of _kernel_dict
        averaging.setter adjusts the underlying data
        subscribe an observer by registering it with "averaging_bind_to"

    width : int
        width of the line in pixels

    length : int or float
        length of the line
        length.setter adjusts the underlying data
        subscribe an observer by registering it with "length_bind_to"

    color : str
        one of "get_color_names" from vispy.color

    conector : str or np.ndarray
        way to render line between coordinates
        two built in types: "segment" or "strip"
        third type is np.ndarray

    mode : str
        control panel mode

    Attributes
    ----------
    Private attributes
    -------
    _kernel_dict
    _kernel
    _data_types
    _data_type
    _width
    _color
    _colors
    _connector_types
    _connector
    _avg_dims
    _averaging
    _length
    _avg_observers
    _len_observers
    _need_display_update
    _need_visual_update
    _raw_dat
    _original_data
    _current_shape
    _vectors
    _mode
    _mode_history

    Public attributes
    -------
    name


    See vispy's line visual docs for more details:
    http://api.vispy.org/en/latest/visuals.html#vispy.visuals.LineVisual
    http://vispy.org/visuals.html
    """

    def __init__(self,
                 vectors,
                 width=1,
                 color='red',
                 averaging='1x1',
                 length=1,
                 name=None):

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
        self._data_types = ('image', 'coords')
        self._data_type = None

        # Save the line style params
        self._width = width
        self._color = color
        self._colors = get_color_names()
        self._connector_types = ['segments']
        self._connector = self._connector_types[0]

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
        self._current_shape = vectors.shape
        self._vectors = self._convert_to_vector_type(vectors)
        # self.vectors = vectors
        self.averaging_bind_to(self._default_avg)
        self.length_bind_to(self._default_length)

        if name is None:
            self.name = 'vectors'
        else:
            self.name = name
        # self.events.add(mode=Event)
        self._qt_properties = QtVectorsLayer(self)

    # ====================== Property getter and setters =====================
    @property
    def _original_data(self) -> np.ndarray:
        return self._raw_dat

    @_original_data.setter
    def _original_data(self, dat: np.ndarray):
        """
        Must preserve data used at construction. Specifically for default
            averaging/length adjustments
        averaging/length adjustments recalculate the underlying data
        :param dat: updated only at construction
        :return:
        """
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
                where x-y are position (center) and u-v are x-y projections of
                    the vector
            2) (N, M, 2) array with elements (u, v)
                where u-v are x-y projections of the vector
                vector position is one per-pixel in the NxM array
        :param vectors: ndarray
        :return:
        """
        self._original_data = vectors

        self._vectors = self._convert_to_vector_type(vectors)

        self.viewer._child_layer_changed = True
        self._refresh()

    def _convert_to_vector_type(self, vectors):
        """
        Check on input data for proper shape and dtype
        :param vectors: ndarray
        :return:
        """
        if vectors.shape[-1] == 4 and len(vectors.shape) == 2:
            self._current_shape = vectors.shape
            coord_list = self._convert_coords_to_coordinates(vectors)
            self._data_type = self._data_types[1]

        elif vectors.shape[-1] == 2 and len(vectors.shape) == 3:
            self._current_shape = vectors.shape
            coord_list = self._convert_image_to_coordinates(vectors)
            self._data_type = self._data_types[0]

        else:
            raise InvalidDataFormatError(
                "Vector data of shape %s is not supported" %
                str(vectors.shape))

        return coord_list

    def _convert_image_to_coordinates(self, vect) -> np.ndarray:
        """
        To convert an image-like array with elements (x-proj, y-proj) into a
            position list of coordinates
        Every pixel position (n, m) results in two output coordinates of (N,2)

        :param vect: ndarray of shape (N, M, 2)
        :return: position list of shape (2*N*M, 2) for vispy
        """
        xdim = vect.shape[0]
        ydim = vect.shape[1]

        # stride is used during averaging and length adjustment
        stride_x = self._kernel[0]
        stride_y = self._kernel[1]

        # create empty vector of necessary shape
        # every "pixel" has 2 coordinates
        pos = np.empty((2 * xdim * ydim, 2), dtype=np.float32)

        # create coordinate spacing for x-y
        # double the num of elements by doubling x sampling
        xspace = np.linspace(0, stride_x*xdim, 2 * xdim, endpoint=False)
        yspace = np.linspace(0, stride_y*ydim, ydim, endpoint=False)
        xv, yv = np.meshgrid(xspace, yspace)

        # assign coordinates (pos) to all pixels
        pos[:, 0] = xv.flatten()
        pos[:, 1] = yv.flatten()

        # pixel midpoints are the first x-values of positions
        midpt = np.zeros((xdim * ydim, 2), dtype=np.float32)
        midpt[:, 0] = pos[0::2, 0]+(stride_x-1)/2
        midpt[:, 1] = pos[0::2, 1]+(stride_y-1)/2

        # rotate coordinates about midpoint to represent angle and length
        pos[0::2, 0] = midpt[:, 0] - (stride_x / 2) * (self._length/2) * \
                       vect.reshape((xdim*ydim, 2))[:, 0]
        pos[0::2, 1] = midpt[:, 1] - (stride_y / 2) * (self._length/2) * \
                       vect.reshape((xdim*ydim, 2))[:, 1]
        pos[1::2, 0] = midpt[:, 0] + (stride_x / 2) * (self._length/2) * \
                       vect.reshape((xdim*ydim, 2))[:, 0]
        pos[1::2, 1] = midpt[:, 1] + (stride_y / 2) * (self._length/2) * \
                       vect.reshape((xdim*ydim, 2))[:, 1]


        return pos

    def _convert_coords_to_coordinates(self, vect) -> np.ndarray:
        """
        To convert a list of coordinates of shape
            (x-center, y-center, x-proj, y-proj) into a list of coordinates
        Input coordinate of (N,4) becomes two output coordinates of (N,2)

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
    def averaging(self) -> str:
        """
        Set the kernel over which to average
        :return: string of averaging kernel size
        """
        return self._averaging
    
    @averaging.setter
    def averaging(self, averaging: str):
        """
        Calculates an average vector over a kernel
        :param averaging: one of "_avg_dims" above
        :return:
        """
        self._averaging = averaging
        self._kernel = self._kernel_dict[averaging]

        if self._default_avg in self._avg_observers:
            self._default_avg(averaging)
        else:
            for callback in self._avg_observers:
                callback(self._averaging)

        self._refresh()

    def averaging_bind_to(self, callback):
        """
        register an observer to be notified upon changes to averaging
        Removes the default method for averaging if an external method
            is registered
        :param callback: function to call upon averaging changes
        :return:
        """
        if self._default_avg in self._avg_observers:
            self._avg_observers.remove(self._default_avg)
        self._avg_observers.append(callback)

    def _default_avg(self, avg_kernel: str):
        """
        Default method for calculating average
        Implemented ONLY for image-like vector data
        :param avg_kernel: kernel over which to compute average
        :return:
        """
        if self._data_type == 'coords':
            # default averaging is supported only for 'matrix' dataTypes
            return None
        elif self._data_type == 'image':
            self._kernel = self._kernel_dict[avg_kernel]

            if self._kernel == (1, 1):
                self.vectors = self._original_data
                return None

            tempdat = self._original_data
            range_x = tempdat.shape[0]
            range_y = tempdat.shape[1]
            x = self._kernel[0]
            y = self._kernel[1]
            x_offset = int((x - 1) / 2)
            y_offset = int((y - 1) / 2)

            kernel = np.ones(shape=(x, y)) / (x*y)

            output_mat = np.zeros_like(tempdat)
            output_mat_x = signal.convolve2d(tempdat[:, :, 0], kernel, mode='same', boundary='wrap')
            output_mat_y = signal.convolve2d(tempdat[:, :, 1], kernel, mode='same', boundary='wrap')

            output_mat[:, :, 0] = output_mat_x
            output_mat[:, :, 1] = output_mat_y

            self.vectors = output_mat[x_offset:range_x-x_offset:x, y_offset:range_y-y_offset:y]

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
        Change the length of all lines
        :param length: length multiplicative factor
        :return: None
        """
        self._length = length

        if self._default_length in self._len_observers:
            self._default_length(length)
        else:
            for callback in self._len_observers:
                callback(self._length)

        self._refresh()

    def length_bind_to(self, callback):
        """
        register an observer to be notified upon changes to length
        Removes the default method for length if an external method is registered
        :param callback: function to call upon length changes
        :return:
        """
        if self._default_length in self._len_observers:
            self._len_observers.remove(self._default_length)
            print('removing default length')
        self._len_observers.append(callback)

    def _default_length(self, newlen: int):
        """
        Default method for calculating vector lengths
        Implemented ONLY for image-like vector data
        :param newlen: new length
        :return:
        """

        if self._data_type == 'coords':
            return None
        elif self._data_type == 'image':
            self._length = newlen
            self._vectors = self._convert_to_vector_type(self._original_data)

    @property
    def color(self) -> str:
        """Color, ColorArray: color of the body of the marker
        """
        return self._color

    @color.setter
    def color(self, color: str) -> None:
        self._color = color
        self._refresh()

    # =========================== Napari Layer ABC methods ===================
    @property
    def data(self) -> np.ndarray:
        """

        :return: coordinates of line vertices
        """
        return self.vectors

    @data.setter
    def data(self, data: np.ndarray) -> None:
        """
        :param data:
        :return:
        """

        self.vectors = data

    def _get_shape(self):
        if len(self.vectors) == 0:
            return np.ones(self.vectors.shape, dtype=int)
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
        
        in_slice_vectors = self.vectors

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
            connect=self._connector)
        self._need_visual_update = True
        self._update()

    # ========================= Napari Layer ABC CONTROL methods =====================


class InvalidDataFormatError(Exception):
    """
    To better describe when vector data is not correct
        more informative than TypeError
    """

    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message
