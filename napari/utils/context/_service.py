"""
Contexts
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from itertools import count
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ChainMap,
    DefaultDict,
    Dict,
    Final,
    MutableMapping,
    Optional,
    Tuple,
)

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

    _key_prefix: Final[str] = 'settings.'

    def __init__(self):
        super().__init__()
        from napari.settings import get_settings

        self._settings = get_settings()
        self._settings.events.changed.connect(self._update_key)

    def _update_key(self, event: Event):
        self.changed.emit(f'{self._key_prefix}{event.key}')

    def __missing__(self, key: str) -> Any:
        if key.startswith(self._key_prefix):
            splits = [k for k in key.split(".")[1:] if k]
            val: Any = self._settings
            if splits:
                while splits:
                    val = getattr(val, splits.pop(0))
                if hasattr(val, 'dict'):
                    val = val.dict()
                return val
        return super().__missing__(key)

    def new_child(self, m: Optional[MutableMapping] = None):
        """New ChainMap with a new map followed by all previous maps.

        If no map is provided, an empty dict is used.
        """
        if m is None:
            m = {}
        return Context(m, self)

    def __setitem__(self, k: str, v: Any) -> None:
        if k.startswith(self._key_prefix):
            raise ValueError(
                f"Cannot set key starting with {self._key_prefix!r}"
            )
        return super().__setitem__(k, v)

    def __del__(self):
        self._settings.events.changed.disconnect(self._update_key)

    def __bool__(self):
        return True


CreateCtx = Callable[[object, int, int, Context], Context]
GetCtx = Callable[[object], Optional[Context]]


def _build_context_closure() -> Tuple[CreateCtx, GetCtx]:
    fmt: str = '{}:{}'

    counter: DefaultDict[str, count[int]] = DefaultDict(count)
    cache: Dict[str, str] = {}
    ROOT_CONTEXT = SettingsAwareContext()
    ALL_CONTEXTS: Dict[Optional[str], Context] = {None: ROOT_CONTEXT}

    def get_scope_id(obj: object) -> str:
        type_ = type(obj).__name__
        cache_key = f'{type_}:{id(obj)}'
        if cache_key not in cache:
            cache[cache_key] = fmt.format(type_, next(counter[type_]))
        return cache[cache_key]

    def create_context(
        obj: object, max_depth=20, start=2, root: Context = ROOT_CONTEXT
    ) -> Context:

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
                    obid = get_scope_id(frame.f_locals['self'])
                    if obid in ALL_CONTEXTS:
                        parent = ALL_CONTEXTS[obid]
                        break
                frame = frame.f_back  # type: ignore
                i += 1

        new_context = parent.new_child()
        ALL_CONTEXTS[get_scope_id(obj)] = new_context
        return new_context

    def get_context(obj: object) -> Optional[Context]:
        return ALL_CONTEXTS.get(get_scope_id(obj))

    return create_context, get_context


create_context, get_context = _build_context_closure()
