from ..base import Layer
from ..utils.layer_utils import combine_extents


class LayerGroup(Layer):
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

    def __len__(self):
        return len(self._children)

    # maybe the number of leaf layers should be an @property on the layergroup
    def total_layers(self):
        return sum([1 for _ in self.traverse_leaf()])

    # you can change __iter__ to be traverse or traverse_leaf with no bad consequences
    def __iter__(self):
        for child in self._children:
            yield child

    def traverse(self):
        "Recursively traverse all nodes and leaves of the LayerGroup."
        yield self
        for child in self:
            if child._children is not None:
                yield from child.traverse()
            else:
                yield child

    def traverse_leaf(self):
        """Recursively traverse LayerGroup tree and yield each leaf node."""
        for child in self:
            if child._children is not None:
                yield from child.traverse_leaf()
            else:
                yield child

    def __getitem__(self, idx):
        return self._children.__getitem__(idx)

    def __setitem__(self, idx, value):
        return self._children.__setitem__(idx, value)

    # We need more & better list handling methods
    # moving a layer from one position to another
    # moving a layer up/down heirarchy branches when moving its position
    # removing a layer from a group (should remove the parent attribute)
    def append(self, item):
        item._parent = self
        return self._children.append(item)

    def _get_extent(self):
        return combine_extents([c._get_extent() for c in self._children])

    def _get_ndim(self):
        return max([c._get_ndim() for c in self._children])

    def _get_state(self):
        """LayerGroup state as a list of state dictionaries.

        Returns
        -------
        state : list
            List of layer state dictionaries.
        """
        state = []
        state.append(self._get_base_state())
        for child in self:
            if child._children is not None:
                state.append(child._get_state())
            else:
                state.append(child._get_state())
        return state

    def _get_value(self):
        # list of all children _get_value()
        # list is nested, is that okay?
        return [c._get_value() for c in self._children]

    def _set_view_slice(self):
        # I think this is properly recursive, double check later when not tired
        for child in self:
            child._set_view_slice()

    def _update_thumbnail(self, *args, **kwargs):
        # we should do something here, leave it for later
        pass

    @property
    def data(self):
        return None

    @data.setter
    def data(self):
        raise NotImplementedError
