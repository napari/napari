from .layerlist import LayerList
from ..layers import Layer


class LayerGroup(Layer):
    def __init__(
        self, iterable=(), *, name='LayerGroup', ndim=2, visible=True
    ) -> None:
        super().__init__(None, ndim)
        self._name = name
        self._children = LayerList(iterable=iterable)

    def traverse(self):
        """Recursively traverse LayerGroup tree and yield each leaf node."""
        for child in self:
            if child._children is not None:
                yield from child.traverse()
            else:
                yield child

    def _render(self):
        """Recursively return list of strings that can render ascii tree."""
        lines = []
        lines.append(self.name)

        for n, child in enumerate(self):
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

        for n, child in enumerate(self):
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

    def __iter__(self):
        for child in self._children:
            yield child

    def __getitem__(self, idx):
        return self._children.__getitem__(idx)

    def __setitem__(self, idx, value):
        return self._children.__setitem__(idx, value)

    def append(self, item):
        return self._children.append(item)

    def _get_extent(self, *args, **kwargs):
        raise NotImplementedError

    def _get_ndim(self, *args, **kwargs):
        raise NotImplementedError

    def _get_state(self, *args, **kwargs):
        raise NotImplementedError

    def _get_value(self, *args, **kwargs):
        raise NotImplementedError

    def _set_view_slice(self, *args, **kwargs):
        raise NotImplementedError

    def _update_thumbnail(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def data(self):
        raise NotImplementedError

    @data.setter
    def data(self):
        raise NotImplementedError

    @property
    def visible(self):
        """bool: Whether the visual is currently being displayed."""
        return self._visible

    @visible.setter
    def visible(self, visibility):
        self._visible = visibility
        for child in self:
            child.visible = visibility
