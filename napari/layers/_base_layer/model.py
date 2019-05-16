from abc import ABC, abstractmethod
from contextlib import contextmanager
from xml.etree.ElementTree import Element, tostring
import numpy as np

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
    indices

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
        self._indices = ()
        self._cursor_position = (0, 0)
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
    def indices(self):
        """Tuple of int of Slice: Used for slicing arrays on each dimension.
        """
        return self._indices

    @indices.setter
    def indices(self, indices):
        if indices == self.indices:
            return
        self._indices = indices[-self.ndim:]
        self._set_view_slice()

    @property
    def coordinates(self):
        """Tuple of float: Coordinates of the cursor in the respective image
        space of each layer.

        The setter expects the a 2-tuple of coordinates in canvas space
        ordered (x, y) and then transforms them to image space and inserts
        them into the correct position of the layer indices. The length of the
        tuple is equal to the number of dimensions of the layer.
        """
        return self._coordinates

    @coordinates.setter
    def coordinates(self, cursor_position):
        transform = self._node.canvas.scene.node_transform(self._node)
        position = tuple(transform.map(cursor_position)[:2])
        coords = list(self.indices)
        coords[-2] = position[1]
        coords[-1] = position[0]
        self._coordinates = tuple(coords)

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
    def range(self):
        """list of 3-tuple of int: ranges of data for slicing specifed by
        (min, max, step).
        """
        shape = self._get_shape()
        return [(0, max, 1) for max in shape]

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

    @property
    def scale_factor(self):
        """float: Conversion factor from canvas coordinates to image
        coordinates, which depends on the current zoom level.
        """
        transform = self._node.canvas.scene.node_transform(self._node)
        scale_factor = transform.map([1, 1])[:2] - transform.map([0, 0])[:2]

        return scale_factor[0]

    def _after_set_viewer(self, prev):
        """Triggered after a new viewer is set.

        Parameters
        ----------
        prev : Viewer
            Previous viewer.
        """
        if self.viewer is not None:
            self.refresh()

    def _update(self):
        """Update the underlying visual."""
        if self._need_display_update:
            self._need_display_update = False
            if hasattr(self._node, '_need_colortransform_update'):
                self._node._need_colortransform_update = True
            self._set_view_slice()

        if self._need_visual_update:
            self._need_visual_update = False
            self._node.update()

    @abstractmethod
    def _set_view_slice(self):
        raise NotImplementedError()

    def refresh(self):
        """Fully refreshes the layer. If layer is frozen refresh will not occur
        """
        if self._freeze:
            return
        self._refresh()

    def _refresh(self):
        """Fully refresh the underlying visual.
        """
        self._need_display_update = True
        self._update()

    @contextmanager
    def freeze_refresh(self):
        self._freeze = True
        yield
        self._freeze = False

    def to_xml_list(self):
        """Generates a list of xml elements for the layer.

        Returns
        ----------
        xml : list of xml.etree.ElementTree.Element
            List of a single xml element specifying the currently viewed image
            as a png according to the svg specification.
        """
        return []

    def to_svg(self, file=None, canvas_shape=None):
        """Returns an svg string with all the currently viewed image as a png
        or writes to svg to a file.

        Parameters
        ----------
        file : path-like object, optional
            An object representing a file system path. A path-like object is
            either a str or bytes object representing a path, or an object
            implementing the `os.PathLike` protocol. If passed the svg will be
            written to this file
        view_box : 4-tuple, optional
            View box of SVG canvas to be generated specified as `min-x`,
            `min-y`, `width` and `height`. If not specified, calculated
            from the last two dimensions of the layer.

        Returns
        ----------
        svg : string
            String with the svg specification of the currently viewed layers
        """

        if view_box is None:
            min_shape = [0, 0]
            max_shape = self.shape[-2:]
            range = np.array(max_shape) - np.array(min_shape)
            view_box = min_shape[::-1] + list(range)[::-1]

        props = {'xmlns': 'http://www.w3.org/2000/svg',
                 'xmlns:xlink': 'http://www.w3.org/1999/xlink'}
        xml = Element('svg', width=f'{canvas_shape[0]}',
                      height=f'{canvas_shape[1]}', version='1.1',
                      **props)

        xml_list = self.to_xml_list()

        for x in xml_list:
            xml.append(x)

        svg = ('<?xml version=\"1.0\" standalone=\"no\"?>\n' +
               '<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n' +
               '\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n' +
               tostring(xml, encoding='unicode', method='xml'))

        if file:
            # Save svg to file
            with open(file, 'w') as f:
                f.write(svg)

        return svg

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
