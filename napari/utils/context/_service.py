"""
Classes that implement scoped contexts: dict-like objects that manage a set
of key value pairs, inheriting from "parent" contexts when a value is missing.

In most cases, `create_context` and `get_context` will be used to access the
functionality here:

Examples
--------

    from napari.utils.context._service import get_context, create_context

    class A:
        def __init__(self) -> None:
            self.ctx = create_context(self)
            self.b = B()  # self.b.ctx will be scoped off of self.ctx

    class B:
        def __init__(self) -> None:
            self.ctx = create_context(self)

    obj = A()
    assert get_context(obj) is obj.ctx

This module is modeled after vscode's context key service

https://github.com/microsoft/vscode/blob/main/src/vs/platform/contextkey/browser/contextKeyService.ts
"""
from __future__ import annotations

import sys
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from itertools import count
from typing import Any, Callable, Dict, Optional, TypeVar

from typing_extensions import Final

from napari.utils.events import EventEmitter


class Context(dict):
    """A (hashable) dict that has a parent.

    If `__getitem__` or `get` can't find a key, the parent is checked.
    use `collect()` to get the merged dict (self takes precedence)
    """

    _parent: Optional[Context] = None

    def __missing__(self, name: str) -> Any:
        return (self._parent or {})[name]  # type: ignore

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        return (self._parent or {}).get(key, default)  # type: ignore

    def collect(self) -> dict:
        """Collect all values including those provided by parent(s)"""
        p = self._parent.collect() if self._parent is not None else {}
        return {**p, **self}

    def _create_child(self, *a, **k) -> Context:
        ctx = Context(*a, **k)
        ctx._parent = self
        return ctx

    def __repr__(self) -> str:
        myvals = dict(self)
        parvals = {k: v for k, v in self.collect().items() if k not in self}
        parstring = f' # <{parvals}>' if parvals else ''
        return f'{type(self).__name__}({myvals}){parstring}'

    def __hash__(self) -> int:  # type: ignore
        return id(self)


class SettingsAwareContext(Context):
    """A special context that allows access of settings using `settings.`"""

    _key_prefix: Final[str] = 'settings.'

    def __init__(self, id: int, emitter: EventEmitter):
        super().__init__()
        self._id = id
        self._emitter = emitter
        from napari.settings import get_settings

        self._settings = get_settings()
        self._settings.events.changed.connect(self._update_key)

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

    def _update_key(self, event):
        ctx_key = f'{self._key_prefix}{event.key}'
        # if ctx_key in self:
        self._emitter(value=[ctx_key])

    def __del__(self):
        self._settings.events.changed.disconnect(self._update_key)

    def __bool__(self):
        return True


class _BaseContextKeyService(ABC):
    _parent: Optional[_BaseContextKeyService] = None
    _scopes: Dict[str, ScopedContextKeyService] = {}

    # this is how things subscribe to changes
    context_changed: EventEmitter

    def __init__(self, id: int) -> None:
        self._context_id = id
        self.context_changed = EventEmitter(self, 'context_changed')

    @contextmanager
    def buffered_change_events(self):
        # used when you want to make a number of changes
        # but only emit a single change event at the end
        ...  # TODO

    def create_scoped(self, obj: object) -> ScopedContextKeyService:
        return ScopedContextKeyService(self, obj)

    # vscode: "getContextKeyValue" (used frequently outside the module)
    def __getitem__(self, key: str) -> Any:
        return self._my_context[key]

    # vscode: "setContext" (only used in this module)
    def __setitem__(self, key: str, value: Any) -> None:
        if self._my_context.get(key, '__missing__') != value:
            self._my_context[key] = value
            self.context_changed(value=[key])

    # vscode: "removeContext" (only used in ContextKey.reset())
    def __delitem__(self, key: str) -> None:
        if key in self._my_context:
            del self._my_context[key]
            self.context_changed(value=[key])

    def get_context(self, obj: object) -> Optional[ScopedContextKeyService]:
        """Return the context for an object (set with create_scoped)."""
        service = self._scopes.get(scope_id(obj))
        if service is not None:
            return service
        return None

    @property
    def _my_context(self) -> Context:
        """Return context for this service."""
        return self._get_context_container(self._context_id)

    # vscode: "getContextValuesContainer" (only used in this module)
    @abstractmethod
    def _get_context_container(self, context_id: int) -> Context:
        """Get Context dict for `context_id`"""

    @abstractmethod
    def _del_context(self, context_id: int) -> None:
        """delete Context dict for `context_id`"""

    # vscode: "createChildContext" (only used in this module when scoping)
    @abstractmethod
    def create_child_context(self, parent_ctx_id: Optional[int] = None) -> int:
        """Create a new child context for `parent_ctx_id`.

        If not provided parent_ctx_id should be self._context_id
        """

    # not present in vscode ... just for dict-like interface
    def __contains__(self, key):
        return key in self._my_context

    def __iter__(self):
        yield from self._my_context.collect().items()

    def __len__(self):
        return len(self._my_context.collect())

    def __repr__(self):
        return f'{type(self).__name__}({dict(self._my_context)})'


class ContextKeyService(_BaseContextKeyService):
    """A root context key service."""

    _contexts: Dict[int, Context]
    __root: Optional[ContextKeyService] = None

    def __init__(self) -> None:
        self._ctx_id_count: count[int] = count(0)
        super().__init__(next(self._ctx_id_count))
        my_ctx = SettingsAwareContext(self._context_id, self.context_changed)
        self._contexts = {self._context_id: my_ctx}

    def _get_context_container(self, context_id: int) -> Context:
        return self._contexts.get(context_id, Context())  # Null context

    def _del_context(self, context_id: int) -> None:
        self._contexts.pop(context_id, None)

    def create_child_context(self, parent_ctx_id: Optional[int] = None) -> int:
        _id = next(self._ctx_id_count)
        pid = parent_ctx_id or self._context_id
        if pid in self._contexts:
            self._contexts[_id] = self._contexts[pid]._create_child()
        else:
            self._contexts[_id] = Context()
        return _id

    @classmethod
    def root(cls) -> ContextKeyService:
        """A (global) root context off of which things can be scoped.

        use ContextKeyService.root().create_scoped(...)
        """
        if cls.__root is None:
            cls.__root = cls()
        return cls.__root


class ScopedContextKeyService(_BaseContextKeyService):
    """A child context of ContextKeyService, scoped on some object."""

    _parent: _BaseContextKeyService

    def __init__(self, parent: _BaseContextKeyService, obj: object) -> None:
        super().__init__(parent.create_child_context())
        self._parent = parent
        self._parent.context_changed.connect(self._reemit_event)

        if obj is None:
            return  # pragma: no cover

        self._scope_id = scope_id(obj)
        if self._scope_id in self._parent._scopes:
            raise RuntimeError(f"{obj} already has a context scope")

        self[type(obj).__name__] = scope_id(obj)
        self._parent._scopes[self._scope_id] = self

        # of the object gets deleted, clean this service up from parent
        weakref.finalize(obj, self._parent._del_context, self._context_id)
        weakref.finalize(obj, self._parent._scopes.pop, self._scope_id, None)

    def __del__(self):
        """Delete this scoped service, cleanup parent, remove attr"""
        if self._parent is not None:
            self._parent.context_changed.disconnect(self._reemit_event)
            self._parent._del_context(self._context_id)
            self._parent._scopes.pop(self._scope_id, None)

    # aka: getContextValuesContainer
    def _get_context_container(self, context_id: int) -> Context:
        return self._parent._get_context_container(context_id)

    def _del_context(self, context_id: int) -> None:
        self._parent._del_context(context_id)

    def create_child_context(self, parent_ctx_id: Optional[int] = None) -> int:
        pid = parent_ctx_id or self._context_id
        return self._parent.create_child_context(pid)

    def _reemit_event(self, event):
        self.context_changed(value=event.value)


def _make_scope_id_func(fmt='{}:{}') -> Callable[[object], str]:
    from itertools import count
    from typing import DefaultDict

    counter: DefaultDict[str, count[int]] = DefaultDict(count)
    cache: Dict[str, str] = {}

    def inner(obj: object) -> str:
        type_ = type(obj).__name__
        cache_key = f'{type_}:{id(obj)}'
        if cache_key not in cache:
            cache[cache_key] = fmt.format(type_, next(counter[type_]))
        return cache[cache_key]

    return inner


scope_id = _make_scope_id_func()


def create_context(obj: object, depth=20, start=2) -> ScopedContextKeyService:
    root_scope = ContextKeyService.root()

    if not hasattr(sys, '_getframe'):  # pragma: no cover
        # we can't inspect stack...
        return root_scope.create_scoped(obj)

    frame = sys._getframe(start)
    i = 0
    while frame and i < depth:
        if frame.f_code.co_name in ('__new__', '_set_default_and_type'):
            # the type is being declared and pydantic is checking defaults
            break  # pragma: no cover
        elif 'self' in frame.f_locals:
            obid = scope_id(frame.f_locals['self'])
            if obid in root_scope._scopes:
                par = root_scope._scopes[obid]
                return par.create_scoped(obj)
        frame = frame.f_back  # type: ignore
        i += 1
    return root_scope.create_scoped(obj)


T = TypeVar("T")


def get_context(obj: object) -> Optional[ScopedContextKeyService]:
    return ContextKeyService.root().get_context(obj)
