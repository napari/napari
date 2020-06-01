from ..base import Layer
from ..utils.layer_utils import combine_extents


class Layergroup(Layer):
    def __init__(
        self, children=None, *, name='LayerGroup', ndim=2, visible=True
    ) -> None:
        super().__init__(None, ndim)
        self._name = name
        if children is None:
            self._children = []
        else:
            self._children = children

    def _render(self):
        """Recursively return list of strings that can render ascii tree."""
        lines = []
        lines.append(self.name)

        for n, child in enumerate(self._children):
            try:
                child_tree = child._render()
            except AttributeError:
                child_tree = [child.name]
            lines.append('  +--' + child_tree.pop(0))
            spacer = '   ' if n == len(self) - 1 else '  |'
            lines.extend([spacer + l for l in child_tree])

        return lines

    def __str__(self):
        """Render ascii tree string representation of this layer group"""
        return "\n".join(self._render())

    def _render_repr(self):
        """Recursively return list of strings for unambiguous representation"""
        lines = []
        cls = type(self)
        lines.append(f"<{cls.__name__} '{self.name}' at {hex(id(self))}>")

        for n, child in enumerate(self._children):
            try:
                child_tree = child._render_repr()
            except AttributeError:
                cls = type(child)
                child_tree = [
                    f"<{cls.__name__} layer '{child.name}' at "
                    f"{hex(id(self))}>"
                ]
            lines.append('  +--' + child_tree.pop(0))
            spacer = '   ' if n == len(self) - 1 else '  |'
            lines.extend([spacer + l for l in child_tree])

        return lines

    def __repr__(self):
        """Render unambiguous tree string representation of this layer group"""
        return "\n".join(self._render_repr())

    def __iter__(self):
        """Iteration yields every non-group layer with a depth first search."""
        for child in self._children:
            yield from child

    def traverse(self):
        "Recursively traverse all nodes and leaves of the LayerGroup tree."
        yield self
        for child in self._children:
            if child._children is not None:
                yield from child.traverse()
            else:
                yield child

    def __getitem__(self, idx):
        """Allows indexing into Layergroup, similar to a list of lists."""
        return self._children.__getitem__(idx)

    # We need more & better list handling methods
    # moving a layer from one position to another
    # moving a layer up/down heirarchy branches when moving its position
    # removing a layer from a group (should remove the parent attribute)
    def append(self, item):
        item._parent = self
        # FIXME - add a check for unique layergroup names in the tree
        # FIXME - update ndim property on layergroup with self._get_ndim()
        return self._children.append(item)

    def __len__(self):
        """Number of all non-group layers contained in the layergroup."""
        return sum([1 for _ in self])

    def _get_extent(self):
        """Combined extent bounding all the individual layergroup layers.

        Returns
        -------
        tuple
            Extent returned as tuple, ndim x 2 (min, max)
        """
        return combine_extents([c._get_extent() for c in self._children])

    def _get_ndim(self):
        try:
            self._ndim = max([c._get_ndim() for c in self._children])
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
        if self._children is not None:
            for layer in self._children:
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
            for child in self._children:
                child.refresh()

    @property
    def data(self):
        return None

    @property
    def blending(self):
        return None

    @blending.setter
    def blending(self, value):
        raise NotImplementedError(
            "You may not set a blending mode on a Layergroup."
        )

    def save(self):
        for layer in self:
            yield layer.save()
