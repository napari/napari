"""
Contexts
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ChainMap, Final, Optional
from weakref import WeakKeyDictionary

from psygnal import Signal

if TYPE_CHECKING:
    from ..events.event import Event

_null = object()


class Context(ChainMap):
    changed = Signal(set)  # the set of keys that changed

    @contextmanager
    def buffered_changes(self):
        with self.changed.paused(lambda a, b: (a[0].union(b[0]),)):
            yield

    def __setitem__(self, k: str, v: Any) -> None:
        emit = self.get(k, _null) is not v
        super().__setitem__(k, v)
        if emit:
            self.changed.emit({k})

    def __delitem__(self, k: str) -> None:
        emit = k in self
        super().__delitem__(k)
        if emit:
            self.changed.emit({k})

    def __hash__(self):
        return id(self)


class SettingsAwareContext(Context):
    """A special context that allows access of settings using `settings.`

    This takes no parents, and will always be a root context.
    """

    _PREFIX: Final[str] = 'settings.'

    def __init__(self):
        super().__init__()
        from napari.settings import get_settings

        self._settings = get_settings()
        self._settings.events.changed.connect(self._update_key)

    def _update_key(self, event: Event):
        self.changed.emit(f'{self._PREFIX}{event.key}')

    def __del__(self):
        self._settings.events.changed.disconnect(self._update_key)

    def __missing__(self, key: str) -> Any:
        if key.startswith(self._PREFIX):
            splits = [k for k in key.split(".")[1:] if k]
            val: Any = self._settings
            if splits:
                while splits:
                    val = getattr(val, splits.pop(0))
                if hasattr(val, 'dict'):
                    val = val.dict()
                return val
        return super().__missing__(key)

    def new_child(self, m: Optional[dict] = None):
        """New ChainMap with a new map followed by all previous maps.

        If no map is provided, an empty dict is used.
        """
        if m is None:
            m = {}
        return Context(m, self)  # important to use self, not *self.maps

    def __setitem__(self, k: str, v: Any) -> None:
        if k.startswith(self._PREFIX):
            raise ValueError(f"Cannot set key starting with {self._PREFIX!r}")
        return super().__setitem__(k, v)

    def __bool__(self):
        return True


ROOT_CONTEXT = SettingsAwareContext()
_ALL_CONTEXTS: WeakKeyDictionary[object, Context] = WeakKeyDictionary()


def create_context(
    obj: object,
    max_depth: int = 20,
    start: int = 2,
    root: Context = ROOT_CONTEXT,
) -> Context:
    if root is not ROOT_CONTEXT:
        assert isinstance(root, Context), 'root must be an instance of Context'

    parent = root
    if hasattr(sys, '_getframe'):  # CPython implementation detail
        frame = sys._getframe(start)
        i = 0
        # traverse call stack looking for another object that has a context
        # to scope this new context off of.
        while frame and i < max_depth:
            if frame.f_code.co_name in (
                '__new__',
                '_set_default_and_type',
            ):
                # type is being declared and pydantic is checking defaults
                break
            elif 'self' in frame.f_locals:
                _ctx = _ALL_CONTEXTS.get(frame.f_locals['self'])
                if _ctx is not None:
                    parent = _ctx
                    break
            frame = frame.f_back  # type: ignore
            i += 1

    new_context = parent.new_child()
    _ALL_CONTEXTS[obj] = new_context
    return new_context


def get_context(obj: object) -> Optional[Context]:
    return _ALL_CONTEXTS.get(obj)
