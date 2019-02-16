from abc import ABC, abstractmethod
from contextlib import contextmanager

import weakref

from vispy.util.event import EmitterGroup, Event

from .._visual_wrapper import VisualWrapper


class Layer(VisualWrapper, ABC):
    """Base layer class.

    Must define the following:
        * ``_get_shape()``: called by ``shape`` property
        * ``_refresh()``: called by ``refresh`` method
        * ``data`` property (setter & getter)

    May define the following:
        * ``_set_view_slice(indices)``: called to set currently viewed slice
        * ``_after_set_viewer()``: called after the viewer is set
        * ``_qt_properties``: QtWidget inserted into the layer list GUI
        * ``_qt_controls``: QtWidget inserted into the controls panel GUI

    Attributes
    ----------
    ndim
    shape
    selected
    viewer

    Methods
    -------
    refresh()
        Refresh the current view.
    """
    def __init__(self, central_node):
        super().__init__(central_node)
        self._selected = False
        self._viewer = None
        self._qt_properties = None
        self._qt_controls = None
        self.name = 'layer'
        self._freeze = False
        self.events = EmitterGroup(source=self,
                                   auto_connect=True,
                                   select=Event,
                                   deselect=Event)

    @property
    @abstractmethod
    def data(self):
        # user writes own docstring
        raise NotImplementedError()

    @data.setter
    @abstractmethod
    def data(self, data):
        raise NotImplementedError()

    @abstractmethod
    def _get_shape(self):
        raise NotImplementedError()

    @abstractmethod
    def _refresh(self):
        raise NotImplementedError()

    @property
    def ndim(self):
        """int: Number of dimensions in the data.
        """
        return len(self.shape)

    @property
    def shape(self):
        """tuple of int: Shape of the data.
        """
        return self._get_shape()

    @property
    def selected(self):
        """boolean: Whether this layer is selected or not.
        """
        return self._selected

    @selected.setter
    def selected(self, selected):
        if selected == self.selected:
            return
        self._selected = selected

        if selected:
            self.events.select()
        else:
            self.events.deselect()

    @property
    def viewer(self):
        """Viewer: Parent viewer widget.
        """
        if self._viewer is not None:
            return self._viewer()

    @viewer.setter
    def viewer(self, viewer):
        prev = self.viewer
        if viewer == prev:
            return

        if viewer is None:
            self._viewer = None
            parent = None
        else:
            self._viewer = weakref.ref(viewer)
            parent = viewer._view.scene

        self._parent = parent
        self._after_set_viewer(prev)

    def _after_set_viewer(self, prev):
        """Triggered after a new viewer is set.

        Parameters
        ----------
        prev : Viewer
            Previous viewer.
        """
        if self.viewer is not None:
            self.refresh()

    def _set_view_slice(self, indices):
        """Called whenever the sliders change. Sets the current view given a
        specific slice to view.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices that make up the slice.
        """

    def refresh(self):
        """Fully refreshes the layer. If layer is frozen refresh will not occur
        """
        if self._freeze:
            return
        self._refresh()

    @contextmanager
    def freeze_refresh(self):
        self._freeze = True
        yield
        self._freeze = False

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
        return None, None, ''

    def add(self, position, indices):
        """Adds object at given mouse position and set of indices.
        Parameters
        ----------
        position : sequence of two int
            Position of mouse cursor in canvas.
        indices : sequence of int or slice
            Indices that make up the slice.
        """

    def remove(self, position, indices):
        """Removes object at given mouse position and set of indices.
        Parameters
        ----------
        position : sequence of two int
            Position of mouse cursor in canvas.
        indices : sequence of int or slice
            Indices that make up the slice.
        """

    def move(self, position, indices):
        """Moves object at given mouse position and set of indices.
        Parameters
        ----------
        position : sequence of two int
            Position of mouse cursor in canvas.
        indices : sequence of int or slice
            Indices that make up the slice.
        """
