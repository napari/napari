from __future__ import annotations

from .layerlist import LayerList
from ..layers import Layer
from ..utils.event import Event


class LayerGroup(LayerList, Layer):
    def __init__(
        self, iterable=(), *, name='LayerGroup', visible=True
    ) -> None:
        super().__init__(iterable)
        self._name = name
        self._selected = True
        self._visible = True
        self.events.add(visible=Event, select=Event, deselect=Event)

    @property
    def visible(self):
        """bool: Whether the visual is currently being displayed."""
        return self._visible

    @visible.setter
    def visible(self, visibility):
        self._visible = visibility
        for child in self:
            child._visible = visibility

    def __repr__(self):
        cls = type(self)
        results = []
        for child in self:
            results.append(child.__repr__())
        string_repr = (
            f"<{cls.__name__} layer {repr(self.name)}"
            + f" at {hex(id(self))} containing > [{' + '.join(results)}]"
        )
        return string_repr
