from typing import Union
from collections import Iterable

import numpy as np

from ._base_layer import Layer
from ._register import add_to_viewer
from .._vispy.scene.visuals import Markers as MarkersNode

from ..elements.qt import QtMarkersLayer

@add_to_viewer
class Markers(Layer):
    """Markers layer.

    Parameters
    ----------
    coords : np.ndarray
        coordinates for each marker.

    symbol : str
        symbol to be used as a marker

    size : int, float, np.ndarray, list
        size of the marker. If given as a scalar, all markers are the
        same size. If given as a list/array, size must be the same
        length as coords and sets the marker size for each marker
        in coords (element-wise).

    edge_width : int, float, None
        width of the symbol edge in px
            vispy docs say: "exactly one edge_width and
            edge_width_rel must be supplied"

    edge_width_rel : int, float, None
        width of the marker edge as a fraction of the marker size.

            vispy docs say: "exactly one edge_width and
            edge_width_rel must be supplied"

    edge_color : Color, ColorArray
        color of the marker border

    face_color : Color, ColorArray
        color of the marker body

    scaling : bool
        if True, marker rescales when zooming

    See vispy's marker visual docs for more details:
    http://api.vispy.org/en/latest/visuals.html#vispy.visuals.MarkersVisual



    """

    def __init__(
        self, coords, symbol='o', size=10, edge_width=1,
            edge_width_rel=None, edge_color='black', face_color='white',
            scaling=True):

        visual = MarkersNode()
        super().__init__(visual)

        # Save the marker coordinates
        self._coords = coords

        # Save the marker style params
        self._symbol = symbol
        self._size = size
        self._edge_width = edge_width
        self._edge_width_rel = edge_width_rel
        self._edge_color = edge_color
        self._face_color = face_color
        self._scaling = scaling

        # update flags
        self._need_display_update = False
        self._need_visual_update = False

        self._qt = QtMarkersLayer(self)

    @property
    def coords(self) -> np.ndarray:
        """ndarray: coordinates of the marker centroids
        """
        return self._coords

    @coords.setter
    def coords(self, coords: np.ndarray):
        self._coords = coords

        self.viewer._child_layer_changed = True
        self._refresh()

    @property
    def data(self) -> np.ndarray:
        """ndarray: coordinates of the marker centroids
        """
        return self._coords

    @data.setter
    def data(self, data: np.ndarray) -> None:
        self.coords = data

    @property
    def symbol(self) -> str:
        """ str: marker symbol
        """
        return self._symbol

    @symbol.setter
    def symbol(self, symbol: str) -> None:
        self._symbol = symbol

        self.refresh()

    @property
    def size(self) -> Union[int, float, np.ndarray, list]:
        """float, ndarray: size of the marker symbol in px
        """

        return self._size

    @size.setter
    def size(self, size: Union[int, float, np.ndarray, list]) -> None:

        if np.isscalar(size):
            self._size = size

            self.refresh()

        elif isinstance(size, Iterable):
            assert len(size) == len(self._coords), \
             'If size is a list/array, must be the same length as '\
             'coords'

            if isinstance(size, list):
                self._size = np.asarray(size)

            else:
                self._size = size

            self.refresh()

        else:
            raise TypeError('size should be float or ndarray')

    @property
    def edge_width(self) -> Union[None, int, float]:
        """None, int, float, None: width of the symbol edge in px
        """

        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: Union[None, float]) -> None:
        self._edge_width = edge_width

        self.refresh()

    @property
    def edge_width_rel(self) -> Union[None, int, float]:
        """None, int, float: width of the marker edge as a fraction
            of the marker size.

            vispy docs say: "exactly one edge_width and
            edge_width_rel must be supplied", but I don't know
            what that means... -KY
        """

        return self._edge_width_rel

    @edge_width_rel.setter
    def edge_width_rel(self, edge_width_rel: Union[None, float]) -> None:
        self._edge_width_rel = edge_width_rel

        self.refresh()

    @property
    def edge_color(self) -> str:
        """Color, ColorArray: the marker edge color
        """

        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: str) -> None:
        self._edge_color = edge_color

        self.refresh()

    @property
    def face_color(self) -> str:
        """Color, ColorArray: color of the body of the marker
        """

        return self._face_color

    @face_color.setter
    def face_color(self, face_color: str) -> None:
        self._face_color = face_color

        self.refresh()

    @property
    def scaling(self) -> bool:
        """bool: if True, marker rescales when zooming
        """

        return self._scaling

    @scaling.setter
    def scaling(self, scaling: bool) -> None:
        self._scaling = scaling

        self.refresh()

    def _get_shape(self):

        return np.max(self.coords, axis=0) + 1

    def _update(self):
        """Update the underlying visual.
        """
        if self._need_display_update:
            self._need_display_update = False

            self._set_view_slice(self.viewer.indices)

        if self._need_visual_update:
            self._need_visual_update = False
            self._node.update()

    def _refresh(self):
        """Fully refresh the underlying visual.
        """
        self._need_display_update = True
        self._update()

    def _set_view_slice(self, indices):
        """Sets the view given the indices to slice with.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.
        """

        # Get a list of the coords for the markers in this slice
        coords = self.coords
        matches = np.equal(
            coords[:, 2:],
            np.broadcast_to(indices[2:], (len(coords), len(indices) - 2)))

        matches = np.all(matches, axis=1)

        in_slice_markers = coords[matches, :2]

        # Display markers if there are any in this slice
        if len(in_slice_markers) > 0:
            # Get the marker sizes
            if isinstance(self.size, (list, np.ndarray)):
                sizes = self.size[matches]

            else:
                sizes = self.size

            # Update the markers node
            self._node.visible = True
            self._node.set_data(
                np.array(in_slice_markers) + 0.5,
                size=sizes, edge_width=self.edge_width, symbol=self.symbol,
                edge_width_rel=self.edge_width_rel,
                edge_color=self.edge_color, face_color=self.face_color,
                scaling=self.scaling)

        else:
            self._node.visible = False

        self._need_visual_update = True
        self._update()
