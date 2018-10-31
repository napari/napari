import weakref

import numpy as np

from ._base import Layer
from .._vispy.scene.visuals import Markers as MarkersNode

from ._register import add_to_viewer


@add_to_viewer
class Markers(Layer):
    """Markers layer.

    Parameters
    ----------
    marker_coords : np.ndarray
        coordinates for each marker.


    """

    def __init__(self, marker_coords, symbol='o', size=10, edge_width=1,
        edge_width_rel=None, edge_color='black', face_color='white',
        scaling=True):

        visual = MarkersNode()
        super().__init__(visual)

        # Save the marker coordinates
        self._marker_coords = marker_coords

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

    @property
    def marker_coords(self):
        """ndarray: coordinates of the marker centroids
        """
        return self._marker_coords

    @property
    def data(self):
        """ndarray: coordinates of the marker centroids
        """
        return self._marker_coords

    @data.setter
    def data(self, data):
        self._marker_coords = data

        self.viewer._child_layer_changed = True
        self.viewer._update()


        self.refresh()

    @property
    def symbol(self):
        """ str: marker symbol
        """
        return self._symbol

    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol
        self.refresh()

    @property
    def size(self):
        """float, ndarray: size of the marker symbol in px
        """

        return self._size

    @size.setter
    def size(self, size):

        if isinstance(size, (float, np.ndarray)):
            self._size = size
            self.refresh()

        else:
            raise TypeError('size should be float or ndarray')

    @property
    def edge_width(self):
        """float, None: width of the symbol edge in px
        """

        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width):
        self._edge_width = edge_width
        self.refresh()

    @property
    def edge_width_rel(self):
        """float, None: width of the marker edge as a fraction
            of the marker size.

            vispy docs say: "exactly one edge_width and
            edge_width_rel must be supplied", but I don't know
            what that means... -KY
        """

        return self._edge_width_rel

    @edge_width_rel.setter
    def edge_width_rel(self, edge_width_rel):
        self._edge_width_rel = edge_width_rel
        self.refresh()

    @property
    def edge_color(self):
        """Color, ColorArray: the marker edge color
        """

        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color):
        self._edge_color = edge_color
        self.refresh()

    @property
    def face_color(self):
        """Color, ColorArray: color of the body of the marker
        """

        return self._face_color

    @face_color.setter
    def face_color(self, face_color):
        self._face_color = face_color
        self.refresh()

    @property
    def scaling(self):
        """bool: if True, marker rescales when zooming
        """

        return self._scaling

    @scaling.setter
    def scaling(self, scaling):
        self._scaling = scaling
        self.refresh()

    def _get_shape(self):

        return np.max(self.marker_coords, axis=0) + 1

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
        coords = self.marker_coords
        matches = np.equal(coords[:, 2:],
                   np.broadcast_to(indices[2:], (len(coords), len(indices) - 2)))

        matches = np.all(matches, axis=1)

        in_slice_markers = coords[matches, :2]

        # Display markers if there are any in this slice
        if len(in_slice_markers) > 0:
            # Get the marker sizes
            if isinstance(self.size, np.ndarray):
                sizes = self.size[matches]

            else:
                sizes = self.size

            # Update the markers node
            self._node.visible = True
            self._node.set_data(
                np.array(in_slice_markers) + 0.5,
                size=sizes, edge_width=self._edge_width, symbol=self._symbol,
                edge_width_rel=self._edge_width_rel, edge_color=self._edge_color,
                face_color=self._face_color,
                scaling=self._scaling)

        else:
            self._node.visible = False

        self._need_visual_update = True
        self._update()
