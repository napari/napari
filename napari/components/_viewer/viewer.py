from math import inf

from numpy import clip, integer, ndarray, append, insert, delete, empty
from copy import copy

from ...util.event import EmitterGroup, Event



class Viewer:
    """Viewer object
    """

    def __init__(self):
        super().__init__()
        from .._layers_list import LayersList
        from .._dims import Dims

        self.events = EmitterGroup(source=self,
                                   auto_connect=True,
                                   status=Event,
                                   help=Event,
                                   active_markers=Event)


        self.dims = Dims()

        self.dims.events.axis.connect(self._update_layers)


        self.layers = LayersList(self)

        self._status = 'Ready'
        self._help = ''
        self._cursor = 'standard'
        self._interactive = True
        self._top = None

        ## TODO: this should be eventually removed!
        ## initialised by QtViewer when it is contrsucted by the model
        self._qtviewer = None

    @property
    def status(self):
        """string: Status string
        """
        return self._status

    @status.setter
    def status(self, status):
        if status == self.status:
            return
        self._status = status
        self.events.status(text=self._status)

    @property
    def help(self):
        """string: String that can be displayed to the
        user in the status bar with helpful usage tips.
        """
        return self._help

    @help.setter
    def help(self, help):
        if help == self.help:
            return
        self._help = help
        self.events.help(text=self._help)

    @property
    def interactive(self):
        """bool: Determines if canvas pan/zoom interactivity is enabled or not.
        """
        return self._interactive

    @interactive.setter
    def interactive(self, interactive):
        if interactive == self.interactive:
            return
        self._view.interactive = interactive
        self._interactive = interactive

    @property
    def cursor(self):
        """string: String identifying cursor displayed over canvas.
        """
        return self._cursor

    @cursor.setter
    def cursor(self, cursor):
        if cursor == self.cursor:
            return
        self._qt.set_cursor(cursor)
        self._cursor = cursor

    @property
    def active_markers(self):
        """int: index of active_markers
        """
        return self._active_markers

    @active_markers.setter
    def active_markers(self, active_markers):
        if active_markers == self.active_markers:
            return
        self._active_markers = active_markers
        self.events.active_markers(index=self._active_markers)

    def reset_view(self):
        """Resets the camera's view.
        """
        self._qtviewer.view.camera.set_range()

    def add_layer(self, layer):
        """Adds a layer to the viewer.

        Parameters
        ----------
        layer : Layer
            Layer to add.
        """
        self.layers.append(layer)
        if len(self.layers) == 1:
            self.reset_view()

    def _new_markers(self):
        if self.dims.ndims == 0:
            empty_markers = empty((0, 2))
        else:
            empty_markers = empty((0, self.dims.ndims))
        self.add_markers(empty_markers)

    def _update_layers(self, axis):
        """Updates the contained layers.
        """
        self.dims._set_2d_viewing()
        slices, projections = self.dims.slice_and_project

        for layer in self.layers:
            layer._set_view_specifications(slices, projections)

    def _update_layer_selection(self, event):
        # iteration goes backwards to find top most selected layer if any
        for layer in self.layers[::-1]:
            if layer.selected:
                self._qtviewer.control_panel.display(layer)
                self.status = layer.status
                self.help = layer.help
                self.cursor = layer.cursor
                self.interactive = layer.interactive
                self._top = layer
                break
        else:
            self._qtviewer.control_panel.display(None)
            self.status = 'Ready'
            self.help = ''
            self.cursor = 'standard'
            self.interactive = True
            self._top = None
        self._qtviewer.canvas.native.setFocus()

    def _on_layers_change(self, event):
        self.dims.set_all_ranges(self._calc_layers_ranges())

    def _calc_layers_ranges(self):
        """Calculates the range along each axis from all present layers.
        """

        ndims = self._calc_layers_num_dims()

        ranges = [(inf, -inf, inf)]*ndims

        for layer in self.layers:
            layer_range = layer.range
            ranges = [(min(a,b),max(c,d), min(e,f)) for (a,c,e),(b,d,f) in zip(ranges,layer_range)]

        return ranges

    def _calc_layers_num_dims(self):
        """Calculates the number of maximum dimensions in the contained images.
        """
        max_dims = 0

        for layer in self.layers:
            dims = layer.ndim
            if dims > max_dims:
                max_dims = dims

        return max_dims
