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
from collections import Iterable
from PyQt5.QtCore import pyqtSignal

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


    See vispy's marker visual docs for more details:
    http://api.vispy.org/en/latest/visuals.html#vispy.visuals.LineVisual
    """
    
    line_length = pyqtSignal(float)
    
    def __init__(
        self, vectors,
            width=1,
            color='red',
            connector='segments',
            averaging='1x1',
            length = 10,
            method='agg',
            antialias=True):

        visual = LinesNode()
        super().__init__(visual)

        # Save the line vertex coordinates
        self._vectors = vectors

        # Save the line style params
        self._width = width
        self._color = color
        self._connector = connector
        self._method = method
        self._antialias = antialias

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

    @property
    def vectors(self) -> np.ndarray:
        """

        :return: coordinates of line vertices
        """
        return self._vectors

    @vectors.setter
    def vectors(self, vectors: np.ndarray):
        self._vectors = vectors

        self.viewer._child_layer_changed = True
        self._refresh()

    @property
    def data(self) -> np.ndarray:
        """

        :return: coordinates of line vertices
        """
        return self._vectors

    @data.setter
    def data(self, data: np.ndarray) -> None:
        self.vectors = data

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
        self._refresh()

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

    @property
    def method(self) -> str:
        """
        *** NOT IMPLEMENTED ***
        method used to render the lines.  one of:
            'agg' = anti-grain geometry
            'gl' = openGL
        :return: render method
        """
        return self._method

    @method.setter
    def method(self, method_type: str):
        """
        ** NOT IMPLEMENTED ***
        :param method_type: 
        :return: 
        """
        self._method = method_type
        self._refresh()

    @property
    def antialias(self) -> bool:
        """
        
        :return: 
        """
        return self._antialias

    @antialias.setter
    def antialias(self, antialias_bool: str) -> None:
        """
        
        :param antialias_bool: 
        :return: 
        """
        self._antialias = antialias_bool
        self._refresh()

    def _get_shape(self):
        if len(self.vectors) == 0:
            return np.ones(self.vectors.shape,dtype=int)
        else:
            return np.max(self.vectors, axis=0) + 1

    def _update(self):
        """Update the underlying visual.
        """
        if self._need_display_update:
            self._need_display_update = False

            self._set_view_slice(self.viewer.dimensions.indices)

        if self._need_visual_update:
            self._need_visual_update = False
            self._node.update()

    def _refresh(self):
        """Fully refresh the underlying visual.
        """
        self._need_display_update = True
        self._update()

    def _slice_vectors(self, indices):
        """Determines the slice of vectors given the indices.

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
                np.broadcast_to(indices[2:], (len(vectors), len(indices) - 2)))

            matches = np.all(matches, axis=1)

            in_slice_vectors = vectors[matches, :2]
            return in_slice_vectors, matches
        else:
            return [], []

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
            # Get the marker sizes
            if isinstance(self.width, (list, np.ndarray)):
                sizes = self.width[matches][::-1]

            else:
                sizes = self.width

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
