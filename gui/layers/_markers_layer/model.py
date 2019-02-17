from typing import Union
from collections import Iterable

import numpy as np
from numpy import clip, integer, ndarray, append, insert, delete, empty
from copy import copy

from .._base_layer import Layer
from .._register import add_to_viewer
from ..._vispy.scene.visuals import Markers as MarkersNode
from vispy.visuals import marker_types
from vispy.color import get_color_names

from .view import QtMarkersLayer
from .view import QtMarkersControls


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
        in coords (element-wise). If n_dimensional is True then can be a list
        of length dims or can be an array of shape Nxdims where N is the number
        of markers and dims is the number of dimensions

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

    n_dimensional : bool
        if True, renders markers not just in central plane but also in all
        n dimensions according to specified marker size

    See vispy's marker visual docs for more details:
    http://api.vispy.org/en/latest/visuals.html#vispy.visuals.MarkersVisual



    """

    def __init__(
        self, coords, symbol='o', size=10, edge_width=1,
            edge_width_rel=None, edge_color='black', face_color='white',
            scaling=True, n_dimensional=False):

        visual = MarkersNode()
        super().__init__(visual)

        # Freeze refreshes
        with self.freeze_refresh():
            # Save the marker coordinates
            self._coords = coords

            # Save the marker style params
            self.symbol = symbol
            self.size = size
            self.edge_width = edge_width
            self.edge_width_rel = edge_width_rel
            self.edge_color = edge_color
            self.face_color = face_color
            self.scaling = scaling
            self.n_dimensional = n_dimensional
            self._marker_types = marker_types
            self._colors = get_color_names()
            self._selected_markers = None
            self._mode = 'pan/zoom'
            self._mode_history = 'pan/zoom'
            self._help = ''

            # update flags
            self._need_display_update = False
            self._need_visual_update = False

            self.name = 'markers'
            self._qt_properties = QtMarkersLayer(self)
            self._qt_controls = QtMarkersControls(self)

    @property
    def coords(self) -> np.ndarray:
        """ndarray: coordinates of the marker centroids
        """
        return self._coords

    @coords.setter
    def coords(self, coords: np.ndarray):
        self._coords = coords

        self.viewer._child_layer_changed = True
        self.refresh()

    @property
    def data(self) -> np.ndarray:
        """ndarray: coordinates of the marker centroids
        """
        return self._coords

    @data.setter
    def data(self, data: np.ndarray) -> None:
        self.coords = data

    @property
    def n_dimensional(self) -> str:
        """ bool: if True, renders markers not just in central plane but also in all
        n dimensions according to specified marker size
        """
        return self._n_dimensional

    @n_dimensional.setter
    def n_dimensional(self, n_dimensional: bool) -> None:
        self._n_dimensional = n_dimensional

        self.refresh()

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

        return self._size_original

    @size.setter
    def size(self, size: Union[int, float, np.ndarray, list]) -> None:

        try:
            self._size = np.broadcast_to(size, self._coords.shape)
        except:
            try:
                self._size = np.broadcast_to(size, self._coords.shape[::-1]).T
            except:
                raise ValueError("Size is not compatible for broadcasting")
        self._size_original = size
        self.refresh()

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

    @property
    def mode(self):
        """None, str: Interactive mode
        """
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode == self.mode:
            return
        if (mode=='add') or (mode=='select'):
                self.viewer._qt.view.interactive = False
                self._help = 'hold <space> to pan/zoom'
                self._mode = mode
        elif mode == 'pan/zoom':
            self.viewer._qt.view.interactive = True
            self._help = ''
            self._mode = mode
        else:
            raise ValueError("Mode not recongnized")
        #self.refresh()

    def _get_shape(self):
        if len(self.coords) == 0:
            # when no markers given, return (1,) * dim
            # we use coords.shape[1] as the dimensionality of the image
            return np.ones(self.coords.shape[1], dtype=int)
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

    # def _set_mode(self, mode):
    #     if (mode=='add') or (mode=='select'):
    #         self.mode = mode
    #         self.help = 'hold <space> to pan/zoom'
    #         self.viewer._qt.view.interactive = False
    #     else:
    #         self.viewer._qt.view.interactive = True
    #         self.mode = None
    #         self.help = ''

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
            if self.n_dimensional is True and self.ndim > 2:
                distances = abs(coords[:, 2:] - indices[2:])
                size_array = self._size[:,2:]/2
                matches = np.all(distances <= size_array, axis=1)
                in_slice_markers = coords[matches, :2]
                size_matches = size_array[matches]
                size_matches[size_matches==0] = 1
                scale_per_dim = (size_matches - distances[matches])/size_matches
                scale_per_dim[size_matches==0] = 1
                scale = np.prod(scale_per_dim, axis=1)
                return in_slice_markers, matches, scale
            else:
                matches = np.all(coords[:, 2:]==indices[2:], axis=1)
                in_slice_markers = coords[matches, :2]
                return in_slice_markers, matches, 1
        else:
            return [], [], []

    def _set_selected_markers(self, indices):
        """Determines selected markers selected given indices.

        Parameters
        ----------
        indices : sequence of int
            Indices to check if marker at.
        """
        in_slice_markers, matches, scale = self._slice_markers(indices)

        # Display markers if there are any in this slice
        if len(in_slice_markers) > 0:
            # Get the marker sizes
            size_array = self._size[matches,:2]*np.expand_dims(scale, axis=1)/2
            distances = abs(in_slice_markers - indices[:2])
            in_slice_matches = np.all(distances <= size_array, axis=1)
            indices = np.where(in_slice_matches)[0]
            if len(indices) > 0:
                selection = np.where(matches)[0][indices[-1]]
            else:
                selection = None
        else:
            selection = None

        self._selected_markers = selection

    def _set_view_slice(self, indices):
        """Sets the view given the indices to slice with.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.
        """

        in_slice_markers, matches, scale = self._slice_markers(indices)

        # Display markers if there are any in this slice
        if len(in_slice_markers) > 0:
            # Get the marker sizes
            sizes = (self._size[matches,:2].mean(axis=1)*scale)[::-1]

            # Update the markers node
            data = np.array(in_slice_markers) + 0.5

        else:
            # if no markers in this slice send dummy data
            data = np.empty((0, 2))
            sizes = 0

        self._node.set_data(
            data[::-1], size=sizes, edge_width=self.edge_width,
            symbol=self.symbol, edge_width_rel=self.edge_width_rel,
            edge_color=self.edge_color, face_color=self.face_color,
            scaling=self.scaling)
        self._need_visual_update = True
        self._update()

    def _get_coord(self, position, indices):
        max_shape = self.viewer.dimensions.max_shape
        transform = self._node.canvas.scene.node_transform(self._node)
        pos = transform.map(position)
        pos = [clip(pos[1], 0, max_shape[0]-1), clip(pos[0], 0,
                                                     max_shape[1]-1)]
        coord = copy(indices)
        coord[0] = int(pos[1])
        coord[1] = int(pos[0])
        return coord

    def get_value(self, position, indices):
        """Returns coordinates, values, and a string for a given mouse position
        and set of indices.

        Parameters
        ----------
        position : sequence of two int
            Position of mouse cursor in canvas.
        indices : sequence of int or slice
            Indices that make up the slice.

        Returns
        ----------
        coord : sequence of int
            Position of mouse cursor in data.
        value : int or float or sequence of int or float
            Value of the data at the coord.
        msg : string
            String containing a message that can be used as
            a status update.
        """
        coord = self._get_coord(position, indices)
        self._set_selected_markers(coord)
        value = self._selected_markers
        msg = f'{coord}'
        if value is None:
            pass
        else:
            msg = msg + ', ' + self.name + ', index ' + str(value)
        return coord, value, msg

    def _add(self, coord):
        """Adds object at given mouse position
        and set of indices.
        Parameters
        ----------
        coord : sequence of indices to add marker at
        """
        self._size = append(self._size, [np.repeat(10, self.ndim)], axis=0)
        self.data = append(self.data, [coord], axis=0)
        self._selected_markers = len(self.data)-1

    def _remove(self):
        """Removes selected object if any.
        """
        index = self._selected_markers
        if index is not None:
            self._size = delete(self._size, index, axis=0)
            self.data = delete(self.data, index, axis=0)
            self._selected_markers = None

    def _move(self, coord):
        """Moves object at given mouse position
        and set of indices.
        Parameters
        ----------
        coord : sequence of indices to move marker to
        """
        index = self._selected_markers
        if index is not None:
            self.data[index] = coord
            self.refresh()

    def interact(self, position, indices, dragging=False, shift=False, ctrl=False,
        pressed=False, released=False, moving=False):
        """Highlights object at given mouse position
        and set of indices.
        Parameters
        ----------
        position : sequence of two int
            Position of mouse cursor in canvas.
        indices : sequence of int or slice
            Indices that make up the slice.
        """
        if self.mode is None:
            pass
        elif self.mode == 'select':
            #If in edit mode
            if pressed and ctrl:
                #Delete an existing box if any on control press
                coord = self._get_coord(position, indices)
                self._remove()
            elif moving and dragging:
                #Drag an existing box if any
                coord = self._get_coord(position, indices)
                self._move(coord)
        elif self.mode == 'add':
            #If in addition mode
            if pressed:
                #Add a new box
                coord = self._get_coord(position, indices)
                self._add(coord)
        else:
            pass
