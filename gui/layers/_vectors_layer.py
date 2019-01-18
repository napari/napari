from typing import Union
from collections import Iterable

import numpy as np

from ._base_layer import Layer
from ._register import add_to_viewer
from .._vispy.scene.visuals import Line as LinesNode
# from vispy.visuals import LineVisual
from vispy.color import get_color_names

from ..elements.qt import QtVectorsLayer

@add_to_viewer
class Vectors(Layer):
    """Line layer.

    See vispy's marker visual docs for more details:
    http://api.vispy.org/en/latest/visuals.html#vispy.visuals.MarkersVisual

    """

    def __init__(
        self, coords,
            width=1,
            color='orange',
            connect='segments',
            method='agg',
            antialias=True):

        visual = LinesNode()
        super().__init__(visual)

        # Save the line vertex coordinates
        self._coords = coords

        # Save the line style params
        self._width = width
        self._color = color
        self._connect = connect
        self._method = method
        self._antialias = antialias

        self._connector_types = ['strip', 'segments']
        self._colors = get_color_names()

        # update flags
        self._need_display_update = False
        self._need_visual_update = False

        self.name = 'vectors'
        self._qt = QtVectorsLayer(self)
        self._selected_markers = None

    @property
    def coords(self) -> np.ndarray:
        """

        :return: coordinates of line vertices
        """
        return self._coords

    @coords.setter
    def coords(self, coords: np.ndarray):
        self._coords = coords

        self.viewer._child_layer_changed = True
        self._refresh()

    @property
    def data(self) -> np.ndarray:
        """

        :return: coordinates of line vertices
        """
        return self._coords

    @data.setter
    def data(self, data: np.ndarray) -> None:
        self.coords = data

    @property
    def width(self) -> Union[int, float]:
        """
        width of the line symbol in pixels
            widths greater than 1px only guaranteed to work with "agg" method
        :return:
        """
        return self._width

    @width.setter
    def width(self, width: Union[int, float]) -> None:

        if np.isscalar(width):
            self._width = width

            self.refresh()

        elif isinstance(width, Iterable):
            assert len(width) == len(self._coords), \
             'If size is a list/array, must be the same length as '\
             'coords'

            if isinstance(width, list):
                self._width = np.asarray(width)

            else:
                self._width = width

            self.refresh()

        else:
            raise TypeError('size should be float or ndarray')


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
    def connect(self):
        """
        line connector.  One of _connector.types = "strip", "segments"
        or can receive an ndarray.
        :return: connector parameter
        """
        return self._connect

    @connect.setter
    def connect(self, connector_type: Union[str, np.ndarray]):
        self._connect = connector_type
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
        if len(self.coords) == 0:
            return np.ones(self.coords.shape,dtype=int)
        else:
            return np.max(self.coords, axis=0) + 1

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


    def _set_selected_markers(self, indices):
        """Determines selected markers selected given indices.
        Borrowed from Markers layer

        Parameters
        ----------
        indices : sequence of int
            Indices to check if marker at.
        """
        in_slice_markers, matches = self._slice_markers(indices)

        # Display markers if there are any in this slice
        if len(in_slice_markers) > 0:
            distances = abs(in_slice_markers - np.broadcast_to(indices[:2], (len(in_slice_markers),2)))
            # Get the marker sizes
            if isinstance(self.width, (list, np.ndarray)):
                widths = self.width[matches]
            else:
                widths = self.width
            matches = np.where(matches)[0]
            in_slice_matches = np.less_equal(distances, np.broadcast_to(widths/2, (2, len(in_slice_markers))).T)
            in_slice_matches = np.all(in_slice_matches, axis=1)
            indices = np.where(in_slice_matches)[0]
            if len(indices) > 0:
                matches = matches[indices[-1]]
            else:
                matches = None
        else:
            matches = None

        self._selected_markers = matches

    def _slice_markers(self, indices):
        """Determines the slice of markers given the indices.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.
        """
        # Get a list of the coords for the markers in this slice
        coords = self.coords
        if len(coords) > 0:
            matches = np.equal(
                coords[:, 2:],
                np.broadcast_to(indices[2:], (len(coords), len(indices) - 2)))

            matches = np.all(matches, axis=1)

            in_slice_markers = coords[matches, :2]
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
            data[::-1], width=self.width,
            color=self.color,
            connect=self.connect)
            # method=self.method,
            # antialias=self.antialias)
        self._need_visual_update = True
        self._update()
