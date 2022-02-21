from __future__ import annotations

import sys
from typing import Any, ChainMap, Dict, MutableMapping, Optional
from weakref import finalize

from typing_extensions import Final

from ..events.event import Event, EventEmitter
from ..translations import trans

_null = object()


class Context(ChainMap):
    def __init__(self, *maps: Dict[str, Any]) -> None:
        super().__init__(*maps)
        self.changed = EventEmitter(self, 'changed')

    # this requires psygnal or changes to events.py ...
    # @contextmanager
    # def buffered_changes(self):
    #     with self.changed.paused(lambda a, b: (a[0].union(b[0]),)):
    #         yield

    def __setitem__(self, k: str, v: Any) -> None:
        emit = self.get(k, _null) is not v
        super().__setitem__(k, v)
        if emit:
            self.changed(value={k})

    def __delitem__(self, k: str) -> None:
        emit = k in self
        super().__delitem__(k)
        if emit:
            self.changed(value={k})

    def new_child(self, m: Optional[MutableMapping] = None) -> Context:
        new = super().new_child(m=m)
        self.changed.connect(new.changed)
        return new

    def __hash__(self):
        return id(self)


class SettingsAwareContext(Context):
    """A special context that allows access of settings using `settings.`

    This takes no parents, and will always be a root context.
    """

    _PREFIX: Final[str] = 'settings.'

    def __init__(self):
        super().__init__()
        from ...settings import get_settings

        self._settings = get_settings()
        self._settings.events.changed.connect(self._update_key)

    def _update_key(self, event: Event):
        self.changed(value={f'{self._PREFIX}{event.key}'})

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

    def new_child(self, m: Optional[dict] = None) -> Context:  # type: ignore
        """New ChainMap with a new map followed by all previous maps.

        If no map is provided, an empty dict is used.
        """
        # important to use self, not *self.maps
        return Context(m or {}, self)  # type: ignore

    def __setitem__(self, k: str, v: Any) -> None:
        if k.startswith(self._PREFIX):
            raise ValueError(
                trans._(
                    "Cannot set key starting with {prefix!r}",
                    deferred=True,
                    prefix=self._PREFIX,
                )
            )

        return super().__setitem__(k, v)

    def __bool__(self):
        return True


_ROOT_CONTEXT: Optional[SettingsAwareContext] = None

# note: it seems like WeakKeyDictionary would be a nice match here, but
# it appears that the object somehow isn't initialized "enough" to register
# as the same object in the WeakKeyDictionary later when queried with
# `obj in _OBJ_TO_CONTEXT` ... so instead we use id(obj)
# _OBJ_TO_CONTEXT: WeakKeyDictionary[object, Context] = WeakKeyDictionary()
_OBJ_TO_CONTEXT: Dict[int, Context] = {}


def create_context(
    obj: object,
    max_depth: int = 20,
    start: int = 2,
    root: Optional[Context] = None,
) -> Optional[Context]:

    if root is None:
        global _ROOT_CONTEXT
        if _ROOT_CONTEXT is None:
            _ROOT_CONTEXT = SettingsAwareContext()
        root = _ROOT_CONTEXT
    else:
        assert isinstance(root, Context), trans._(
            'root must be an instance of Context', deferred=True
        )

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
                # this context will never be used.
                return None
            elif 'self' in frame.f_locals:
                _ctx = _OBJ_TO_CONTEXT.get(id(frame.f_locals['self']))
                if _ctx is not None:
                    parent = _ctx
                    break
            frame = frame.f_back  # type: ignore
            i += 1

    new_context = parent.new_child()
    obj_id = id(obj)
    _OBJ_TO_CONTEXT[obj_id] = new_context
    # remove key from dict when object is deleted
    finalize(obj, lambda: _OBJ_TO_CONTEXT.pop(obj_id, None))
    return new_context


def get_context(obj: object) -> Optional[Context]:
    return _OBJ_TO_CONTEXT.get(id(obj))
