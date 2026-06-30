from typing import TYPE_CHECKING, TypeVar

from app_model.expressions import ContextNamespace as _ContextNamespace

if TYPE_CHECKING:
    from napari.utils.events import Event

A = TypeVar('A')


class ContextNamespace[A](_ContextNamespace):
    """A collection of related keys in a context

    meant to be subclassed, with class attributes that are `ContextKeys`.
    """

    def update(self, event: 'Event') -> None:
        """Trigger an update of all "getter" functions in this namespace."""
        for k, get in self._getters.items():
            setattr(self, k, get(event.source))
