import weakref

from typing import Union, Dict

import numpy as np

from ._base import Layer
from .._vispy.scene.visuals import Markers as MarkersNode

from ._register import add_to_viewer


@add_to_viewer
class Marker(Layer):
    """Marker layer.

    Parameters
    ----------
    marker_coords : np.ndarray
        coordinates for each marker.

    marker_args: Dict
        keyword arguments for the marker style. 
        see set_data(): http://api.vispy.org/en/latest/visuals.html#vispy.visuals.MarkersVisual

    """

    def __init__(self, marker_coords, marker_args={}):

        visual = MarkersNode()
        super().__init__(visual)

        # Save the marker coordinates
        self._marker_coords = marker_coords

        self._marker_args = marker_args

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
        self.refresh()

    @property
    def marker_args(self):
        """Dict: keyword arguments for the marker styles
           see set_data(): http://api.vispy.org/en/latest/visuals.html#vispy.visuals.MarkersVisual
        """

        return self._marker_args

    @marker_args.setter
    def marker_args(self, marker_args):
        self._marker_args = marker_args
        self.refresh()

    def _get_shape(self):

        return self.marker_coords.shape

    def _update(self):
        """Update the underlying visual.
        """
        if self._need_display_update:
            self._need_display_update = False

            self.viewer._child_layer_changed = True
            self.viewer._update()

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

        in_slice_markers = []
        indices = list(indices)

        # Get a list of the coords for the markers in this slice
        for coord in self._marker_coords:
            if np.array_equal(coord[2:], indices[2:]):
                in_slice_markers.append(coord[:2])

        # Display markers if there are any in this slice
        if in_slice_markers:
            self._node.visible = True
            self._node.set_data(np.array(in_slice_markers) + 0.5, **self._marker_args)

        else:
            self._node.visible = False


        self._need_visual_update = True
        self._update()
