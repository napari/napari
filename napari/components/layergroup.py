from .layerlist import LayerList
from ..layers import Layer


class LayerGroup(Layer):
    def __init__(
        self, iterable=(), *, name='LayerGroup', ndim=2, visible=True
    ) -> None:
        super().__init__(None, ndim)
        self._name = name
        self._children = LayerList(iterable=iterable)

    def __repr__(self):
        cls = type(self)
        results = []
        for child in self._children:
            results.append(child.__repr__())
        string_repr = (
            f"<{cls.__name__} layer {repr(self.name)}"
            + f" at {hex(id(self))} containing > [{' + '.join(results)}]"
        )
        return string_repr

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
        for child in self._children:
            child._visible = visibility
