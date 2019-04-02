import weakref
from collections.abc import Iterable, Sequence

from ...layers import Layer

from ...util.naming import inc_name_count
from ...util.list import ListModel
from .view import QtLayersPanel


def _add(event):
    """Callback when an item is added to set its order and viewer.
    """
    layers = event.source
    layer = event.item
    layer.name = layers._coerce_name(layer.name, layer)
    layer._order = -len(layers)
    layer.viewer = layers.viewer
    layer.events.select.connect(layers.viewer._update_layer_selection)
    layer.events.deselect.connect(layers.viewer._update_layer_selection)


def _remove(event):
    """Callback when an item is removed to remove its viewer
    and reset its order.
    """
    layers = event.source
    layer = event.item
    layer.viewer = None
    layer._order = 0
    layer.events.select.disconnect(layers.viewer._update_layer_selection)
    layer.events.deselect.disconnect(layers.viewer._update_layer_selection)


def _reorder(event):
    """Callback when the list is reordered to propagate those changes
    to the node draw order.
    """
    layers = event.source
    for i in range(len(layers)):
        layers[i]._order = -i
    canvas = layers.viewer._canvas
    canvas._draw_order.clear()
    canvas.update()


class LayersList(ListModel):
    """List-like layer collection with built-in reordering and callback hooks.

    Parameters
    ----------
    viewer : Viewer, optional
        Parent viewer.

    Attributes
    ----------
    viewer : Viewer
        Parent viewer.
    changed : vispy.util.event.EmitterGroup
        Event hooks:
            * added(item, index): whenever an item is added
            * removed(item): whenever an item is removed
            * reordered(): whenever the list is reordered
    """
    def __init__(self, viewer=None):
        super().__init__(basetype=Layer,
                         lookup={str: lambda q, e: q == e.name})
        self._viewer = None

        self.changed.added.connect(_add)
        self.changed.removed.connect(_remove)
        self.changed.reordered.connect(_reorder)

        # property setting - happens last
        self.viewer = viewer
        self._qt = QtLayersPanel(self)

    def __newlike__(self, iterable):
        return ListModel(self._basetype, iterable, self._lookup)

    @property
    def viewer(self):
        """Viewer: Parent viewer.
        """
        if self._viewer is None:
            return self._viewer

        return self._viewer()

    @viewer.setter
    def viewer(self, viewer):
        prev = self.viewer
        if viewer == prev:
            return

        if prev is not None:
            self.changed.added.disconnect(prev.dims._on_layers_change)
            self.changed.removed.disconnect(prev.dims._on_layers_change)
            self.changed.added.disconnect(prev._update_layer_selection)
            self.changed.removed.disconnect(prev._update_layer_selection)
            self.changed.reordered.disconnect(prev._update_layer_selection)

        for layer in self:
            layer.viewer = viewer

        if viewer is not None:
            self.changed.added.connect(viewer.dims._on_layers_change)
            self.changed.removed.connect(viewer.dims._on_layers_change)
            self.changed.added.connect(viewer._update_layer_selection)
            self.changed.removed.connect(viewer._update_layer_selection)
            self.changed.reordered.connect(viewer._update_layer_selection)
            viewer = weakref.ref(viewer)

        self._viewer = viewer

    def _coerce_name(self, name, layer=None):
        """Coerce a name into a unique equivalent.

        Parameters
        ----------
        name : str
            Original name.
        layer : Layer, optional
            Layer for which name is generated.

        Returns
        -------
        new_name : str
            Coerced, unique name.
        """
        for l in self:
            if l is layer:
                continue
            if l.name == name:
                name = inc_name_count(name)

        return name

    def _move_layers(self, index, insert):
        """Reorder list by moving the item at index and inserting it
        at the insert index. If additional items are selected these will
        get inserted at the insert index too. This allows for rearranging
        the list based on dragging and dropping a selection of items, where
        index is the index of the primary item being dragged, and insert is
        the index of the drop location, and the selection indicates if
        multiple items are being dragged.

        Parameters
        ----------
        index : int
            Index of primary item to be moved
        insert : int
            Index that item(s) will be inserted at
        """
        total = len(self)
        indices = list(range(total))
        if self[index].selected:
            selected = [i for i in range(total) if self[i].selected]
        else:
            selected = [index]
        for i in selected:
            indices.remove(i)
        offset = sum([i < insert for i in selected])
        for insert_idx, elem_idx in enumerate(selected, start=insert - offset):
            indices.insert(insert_idx, elem_idx)
        self[:] = self[tuple(indices)]

    def unselect_all(self, ignore=None):
        """Unselects all layers expect any specified in ignore.

        Parameters
        ----------
        ignore : Layer | None
            Layer that should not be unselected if specified.
        """
        for layer in self:
            if layer.selected and layer != ignore:
                layer.selected = False

    def remove_selected(self):
        """Removes selected items from list.
        """
        to_delete = []
        for i in range(len(self)):
            if self[i].selected:
                to_delete.append(i)
        to_delete.reverse()
        for i in to_delete:
            self.pop(i)
