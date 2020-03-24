from __future__ import annotations

from .layerlist import LayerList
from ..layers import Layer


class LayerGroup(Layer):
    """
    The Composite class represents the complex components that may have
    children. Usually, the Composite objects delegate the actual work to their
    children and then "sum-up" the result.
    """

    def __init__(self, name='LayerGroup', visible=True) -> None:
        self._children: LayerList[Layer] = []
        self._name = name
        self._visible = visible

    """
    A composite object can add or remove other components (both simple or
    complex) to or from its child list.
    """

    def append(self, component: Layer) -> None:
        self._children.append(component)
        component.parent = self

    def remove(self, component: Layer) -> None:
        self._children.remove(component)
        component.parent = None

    @property
    def visible(self):
        """bool: Whether the visual is currently being displayed."""
        return self._visible

    @visible.setter
    def visible(self, visibility):
        self._visible = visibility
        for child in self._children:
            child._visible = visibility

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

    def __getitem__(self, indices):
        return self._children[indices]

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

    def data(self, *args, **kwargs):
        raise NotImplementedError
