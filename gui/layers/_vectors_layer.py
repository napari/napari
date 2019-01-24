from typing import Union

import numpy as np

from ._base_layer import Layer
from ._register import add_to_viewer
from .._vispy.scene.visuals import Line as LinesNode
from vispy.color import get_color_names

from .qt import QtVectorsLayer

from PyQt5.QtCore import pyqtSignal, pyqtSlot

@add_to_viewer
class Vectors(Layer):
    """Line layer.

    """
    # avg_changed = pyqtSignal(str)

    def __init__(
        self, vectors,
            width=1,
            color='red',
            connector='segments',
            averaging='1x1',
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
        self._observers = []

        # update flags
        self._need_display_update = False
        self._need_visual_update = False

        self.name = 'vectors'
        self._qt = QtVectorsLayer(self)
        self._selected_markers = None

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
        
        :return: string of averaging kernel size
        """
        return self._averaging
    
    @averaging.setter
    def averaging(self, averaging: str):
        self._averaging = averaging
        # self.avg_changed.emit(averaging)
        for callback in self._observers:
            print('averaging changed, broadcasting to subscribers')
            callback(self._averaging)

    def bind_to(self, callback):
        print('bound')
        self._observers.append(callback)
        
    @property
    def width(self) -> Union[int, float]:
        """
        width of the line in pixels
            widths greater than 1px only guaranteed to work with "agg" method
        :return:
        """
        return self._width

    @width.setter
    def width(self, width: Union[int, float]) -> None:
        self._width = width
        self.refresh()

    @property
    def color(self) -> str:
        """Color, ColorArray: color of the body of the marker
        """
        return self._color

    @color.setter
    def color(self, color: str) -> None:
        self._color = color
        self.refresh()

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
        self.refresh()

    @property
    def method(self) -> str:
        """
        method used to render the lines.  one of:
            'agg' = anti-grain geometry
            'gl' = openGL
        :return: render method
        """
        return self._method

    @method.setter
    def method(self, method_type: str):
        self._method = method_type
        self.refresh()

    @property
    def antialias(self) -> bool:
        """Color, ColorArray: color of the body of the marker
        """
        return self._antialias

    @antialias.setter
    def antialias(self, antialias_bool: str) -> None:
        self._antialias = antialias_bool
        self.refresh()

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

    # def _set_selected_markers(self, indices):
    #     """Determines selected markers selected given indices.
    #     Borrowed from Markers layer
    #
    #     Parameters
    #     ----------
    #     indices : sequence of int
    #         Indices to check if marker at.
    #     """
    #     in_slice_markers, matches = self._slice_markers(indices)
    #
    #     # Display markers if there are any in this slice
    #     if len(in_slice_markers) > 0:
    #         distances = abs(in_slice_markers - np.broadcast_to(indices[:2], (len(in_slice_markers),2)))
    #         # Get the marker sizes
    #         if isinstance(self.width, (list, np.ndarray)):
    #             widths = self.width[matches]
    #         else:
    #             widths = self.width
    #         matches = np.where(matches)[0]
    #         in_slice_matches = np.less_equal(distances, np.broadcast_to(widths/2, (2, len(in_slice_markers))).T)
    #         in_slice_matches = np.all(in_slice_matches, axis=1)
    #         indices = np.where(in_slice_matches)[0]
    #         if len(indices) > 0:
    #             matches = matches[indices[-1]]
    #         else:
    #             matches = None
    #     else:
    #         matches = None
    #
    #     self._selected_markers = matches

    def _slice_markers(self, indices):
        """Determines the slice of markers given the indices.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.
        """
        # Get a list of the vectors for the markers in this slice
        vectors = self.vectors
        if len(vectors) > 0:
            matches = np.equal(
                vectors[:, 2:],
                np.broadcast_to(indices[2:], (len(vectors), len(indices) - 2)))

            matches = np.all(matches, axis=1)

            in_slice_markers = vectors[matches, :2]
            return in_slice_markers, matches
        else:
            return [], []

    def _set_view_slice(self, indices):
        """Sets the view given the indices to slice with.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.
        """

        in_slice_markers, matches = self._slice_markers(indices)

        # Display markers if there are any in this slice
        if len(in_slice_markers) > 0:
            # Get the marker sizes
            if isinstance(self.width, (list, np.ndarray)):
                sizes = self.width[matches][::-1]

            else:
                sizes = self.width

            # Update the markers node
            data = np.array(in_slice_markers) + 0.5

        else:
            # if no markers in this slice send dummy data
            data = np.empty((0, 2))
            sizes = 0

        self._node.set_data(
            data[::-1],
            width=self.width,
            color=self.color,
            connect=self.connector)
            # method=self.method,
            # antialias=self.antialias)
        self._need_visual_update = True
        self._update()
