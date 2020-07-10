from typing import Iterable

from ...utils.tree import Group
from ..base import Layer
from ..utils.layer_utils import combine_extents
from ...utils.naming import force_unique_name


class LayerGroup(Layer, Group):
    def __init__(
        self, children: Iterable[Layer] = None, name='LayerGroup'
    ) -> None:
        Layer.__init__(self, None, 2, name=name)
        Group.__init__(self, children)
        self.events.inserted.connect(self._on_layer_inserted)

    # TODO: do we need an event for this?  can't this just be in the insert() method?
    def _on_layer_inserted(self, event):
        """When a layer is added, set its name."""
        layer = event.value
        # # Coerce name into being unique in layer list
        # TODO: have discussion about this.  While we do need a unique id
        # for each layer. I'm not convinced that it needs to be the name
        layer.name = force_unique_name(
            layer.name, [lay.name for lay in self if lay != layer]
        )
        # # Unselect all other layers
        self.unselect_all(ignore=event.value)

    # LAYER METHODS

    def _get_extent(self):
        """Combined extent bounding all the individual layergroup layers.

        Returns
        -------
        tuple
            Extent returned as tuple, ndim x 2 (min, max)
        """
        return combine_extents([c._get_extent() for c in self])

    def _get_ndim(self):
        try:
            self._ndim = max([c._get_ndim() for c in self])
        except ValueError:
            self._ndim = 2
        return self._ndim

    def _get_state(self):
        """LayerGroup state as a list of state dictionaries.

        Returns
        -------
        state : list
            List of layer state dictionaries.
        """
        state = []
        state.append(self._get_base_state())
        if self is not None:
            for layer in self:
                state.append(layer._get_state())
        return state

    def _get_value(self):
        """Returns a flat list of all layer values in the layergroup
        for a given mouse position and set of indices.

        Layers in layergroup are iterated over by depth-first recursive search.

        Returns
        ----------
        value : list
            Flat list containing values of the layer data at the coord.
        """
        return [layer._get_value() for layer in self]

    def _set_view_slice(self):
        """Set the view for each layer given the indices to slice with."""
        for child in self:
            child._set_view_slice()

    def _set_highlight(self):
        """Render layer hightlights when appropriate."""
        for child in self:
            child._set_highlight()

    def _update_thumbnail(self, *args, **kwargs):
        # FIXME
        # we should do something here, leave it for later
        pass

    def refresh(self, event=None):
        """Refresh all layer data if visible."""
        if self.visible:
            for child in self:
                child.refresh()

    @property
    def data(self):
        return None

    @property
    def blending(self):
        return None

    @blending.setter
    def blending(self, val):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    # TODO: Selection model

    @property
    def selected_children(self):
        return [l for l in self if l.selected]

    def remove_selected(self):
        """Removes selected items from list."""
        selected = self.selected_children
        [self.remove(i) for i in selected]
        if selected and self:
            # changing 1 -> assuming we won't reverse sort the layerlist anymore
            self[0].selected = True

    def unselect_all(self, ignore=None):
        """Unselects all layers expect any specified in ignore."""
        for layer in self:
            if layer.selected != ignore:
                layer.selected = False


# class ___LayerGroup(EventedList, Layer):
#     def __init__(
#         self, children=None, *, name='LayerGroup', ndim=2, visible=True
#     ) -> None:
#         Layer.__init__(self, None, ndim)
#         EventedList.__init__(self)
#         self._name = name
#         self.extend(children or [])

#     # def _coerce_name(self, name, layer=None):
#     #     return self._coerce_name(name, layer)

#     def _render(self):
#         """Recursively return list of strings that can render ascii tree."""
#         lines = []
#         lines.append(self.name)

#         for n, child in enumerate(self):
#             try:
#                 child_tree = child._render()
#             except AttributeError:
#                 child_tree = [child.name]
#             lines.append('  +--' + child_tree.pop(0))
#             spacer = '   ' if n == len(self) - 1 else '  |'
#             lines.extend([spacer + l for l in child_tree])

#         return lines

#     def __str__(self):
#         """Render ascii tree string representation of this layer group"""
#         return "\n".join(self._render())

#     def _render_repr(self):
#         """Recursively return list of strings for unambiguous representation"""
#         lines = []
#         cls = type(self)
#         lines.append(f"<{cls.__name__} '{self.name}' at {hex(id(self))}>")

#         for n, child in enumerate(self):
#             try:
#                 child_tree = child._render_repr()
#             except AttributeError:
#                 cls = type(child)
#                 child_tree = [
#                     f"<{cls.__name__} layer '{child.name}' at "
#                     f"{hex(id(self))}>"
#                 ]
#             lines.append('  +--' + child_tree.pop(0))
#             spacer = '   ' if n == len(self) - 1 else '  |'
#             lines.extend([spacer + l for l in child_tree])

#         return lines

#     def __repr__(self):
#         """Render unambiguous tree string representation of this layer group"""
#         return "\n".join(self._render_repr())

#     # LIST-LIKE METHODS

#     def iter_layers(self):
#         """Iteration yields every non-group layer with a depth first search."""
#         for child in self:
#             yield from child.iter_layers()

#     def __setitem__(self, key, value):
#         raise NotImplementedError("LayerGroup assignment is not allowed")

#     def __delitem__(self, key):
#         item = self._list.pop(key)
#         item._parent = None
#         if key < 0:
#             # always emit a positive index
#             key += len(self._list) + 1
#         self.events.removed((key, item))

#     def insert(self, index, item):
#         item._parent = self
#         super().insert(index, item)

#     def traverse(self):
#         "Recursively traverse all nodes and leaves of the LayerGroup tree."
#         yield self
#         for child in self:
#             try:
#                 yield from child.traverse()
#             except AttributeError:
#                 yield child

#     @property
#     def selected_children(self):
#         return filter(lambda x: getattr(x, 'selected', False), self)

#     def remove_selected(self):
#         """Removes selected items from list."""
#         selected = list(self.selected_children)
#         [self.remove(i) for i in selected]
#         if selected and self:
#             self[-1].selected = True

#     def select_all(self):
#         """Selects all layers."""
#         for layer in self:
#             if not layer.selected:
#                 layer.selected = True

#     def unselect_all(self, ignore=None):
#         """Unselects all layers expect any specified in ignore."""
#         for layer in self:
#             if layer.selected and layer != ignore:
#                 layer.selected = False

#     def find_id(self, mem_id: int):
#         for item in self.traverse():
#             if id(item) == mem_id:
#                 return item

#     # LAYER-LIKE METHODS

#     def _get_extent(self):
#         """Combined extent bounding all the individual layergroup layers.

#         Returns
#         -------
#         tuple
#             Extent returned as tuple, ndim x 2 (min, max)
#         """
#         return combine_extents([c._get_extent() for c in self])

#     def _get_ndim(self):
#         try:
#             self._ndim = max([c._get_ndim() for c in self])
#         except ValueError:
#             self._ndim = 2
#         return self._ndim

#     def _get_state(self):
#         """LayerGroup state as a list of state dictionaries.

#         Returns
#         -------
#         state : list
#             List of layer state dictionaries.
#         """
#         state = []
#         state.append(self._get_base_state())
#         if self is not None:
#             for layer in self:
#                 state.append(layer._get_state())
#         return state

#     def _get_value(self):
#         """Returns a flat list of all layer values in the layergroup
#         for a given mouse position and set of indices.

#         Layers in layergroup are iterated over by depth-first recursive search.

#         Returns
#         ----------
#         value : list
#             Flat list containing values of the layer data at the coord.
#         """
#         return [layer._get_value() for layer in self]

#     def _set_view_slice(self):
#         """Set the view for each layer given the indices to slice with."""
#         for child in self:
#             child._set_view_slice()

#     def _set_highlight(self):
#         """Render layer hightlights when appropriate."""
#         for child in self:
#             child._set_highlight()

#     def _update_thumbnail(self, *args, **kwargs):
#         # FIXME
#         # we should do something here, leave it for later
#         pass

#     def refresh(self, event=None):
#         """Refresh all layer data if visible."""
#         if self.visible:
#             for child in self:
#                 child.refresh()

#     @property
#     def data(self):
#         return None

#     @property
#     def blending(self):
#         return None

#     def save(self):
#         raise NotImplementedError()
