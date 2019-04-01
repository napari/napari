from abc import ABC, abstractmethod
from contextlib import contextmanager

import weakref

from ...util.event import Event
from ._visual_wrapper import VisualWrapper


class Layer(VisualWrapper, ABC):
    """Base layer class.

    Parameters
    ----------
    central_node : vispy.scene.visuals.VisualNode
        Visual node that controls all others.
    name : str, optional
        Name of the layer. If not provided, is automatically generated
        from `cls._basename()`

    Notes
    -----
    Must define the following:
        * `_get_shape()`: called by `shape` property
        * `_refresh()`: called by `refresh` method
        * `data` property (setter & getter)

    May define the following:
        * `_set_view_slice(indices)`: called to set currently viewed slice
        * `_after_set_viewer()`: called after the viewer is set
        * `_qt_properties`: QtWidget inserted into the layer list GUI
        * `_qt_controls`: QtWidget inserted into the controls panel GUI
        * `_basename()`: base/default name of the layer

    Attributes
    ----------
    name
    ndim
    shape
    selected
    viewer

    Methods
    -------
    refresh()
        Refresh the current view.
    """
    def __init__(self, central_node, name=None):
        super().__init__(central_node)
        self._selected = False
        self._viewer = None
        self._qt_properties = None
        self._qt_controls = None
        self._freeze = False
        self._status = 'Ready'
        self._help = ''
        self._cursor = 'standard'
        self._cursor_size = None
        self._interactive = True
        self.events.add(select=Event,
                        deselect=Event,
                        name=Event)
        self.name = name

    def __str__(self):
        """Return self.name
        """
        return self.name

    def __repr__(self):
        cls = type(self)
        return f"<{cls.__name__} layer {repr(self.name)} at {hex(id(self))}>"

    @classmethod
    def _basename(cls):
        return f'{cls.__name__} 0'

    @property
    def name(self):
        """str: Layer's unique name.
        """
        return self._name

    @name.setter
    def name(self, name):
        if not name:
            name = self._basename()

        if self.viewer:
            name = self.viewer.layers._coerce_name(name, self)

        self._name = name
        self.events.name()

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

    @property
    def status(self):
        """string: Status string
        """
        return self._status

    @status.setter
    def status(self, status):
        if status == self.status:
            return
        self.viewer.status = status
        self._status = status

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
        self.viewer.help = help
        self._help = help

    @property
    def interactive(self):
        """bool: Determines if canvas pan/zoom interactivity is enabled or not.
        """
        return self._interactive

    @interactive.setter
    def interactive(self, interactive):
        if interactive == self.interactive:
            return
        self.viewer.interactive = interactive
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
        self.viewer.cursor = cursor
        self._cursor = cursor

    @property
    def cursor_size(self):
        """int | None: Size of cursor if custom. None is yields default size
        """
        return self._cursor_size

    @cursor_size.setter
    def cursor_size(self, cursor_size):
        if cursor_size == self.cursor_size:
            return
        self.viewer.cursor_size = cursor_size
        self._cursor_size = cursor_size

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

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.
        """
        return

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.
        """
        return

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.
        """
        return

    def on_key_press(self, event):
        """Called whenever key pressed in canvas.
        """
        return

    def on_key_release(self, event):
        """Called whenever key released in canvas.
        """
        return
