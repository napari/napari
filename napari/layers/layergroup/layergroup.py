from __future__ import annotations

from typing import Iterable

from ...utils.tree import Group
from ..base import Layer
from ..utils.layer_utils import combine_extents
from ...utils.naming import inc_name_count


class LayerGroup(Layer, Group):
    def __init__(
        self, children: Iterable[Layer] = (), name: str = 'LayerGroup'
    ) -> None:
        Layer.__init__(self, None, 2)
        Group.__init__(self, children, name=name)

    # LAYER METHODS

    def __str__(self):
        return Group.__str__(self)

    def __repr__(self):
        return Group.__repr__(self)

    def _coerce_name(self, name, layer=None):
        """Coerce a name into a unique equivalent.

        Parameters
        ----------
        name : str
            Original name.
        layer : napari.layers.Layer, optional
            Layer for which name is generated.

        Returns
        -------
        new_name : str
            Coerced, unique name.
        """

        for existing_name in sorted(
            x.name
            for x in self.traverse(with_ancestors=True)
            if x is not layer
        ):
            if name == existing_name:
                name = inc_name_count(name)
        return name

    def _update_name(self, event):
        """Coerce name of the layer in `event.layer`."""
        layer = event.source
        layer.name = self._coerce_name(layer.name, layer)

    def insert(self, index: int, value: Layer):
        """Insert ``value`` before index."""
        new_layer = self._type_check(value)
        new_layer.name = self._coerce_name(new_layer.name)
        super().insert(index, new_layer)

    def _extent_data(self):
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
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
        return [lay for lay in self if lay.selected]

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
