"""
This module is modeled after vscode's context key service

https://github.com/microsoft/vscode/blob/main/src/vs/platform/contextkey/browser/contextKeyService.ts
"""
from __future__ import annotations

import contextlib
import pprint
from abc import ABC, abstractmethod
from contextlib import contextmanager
from itertools import count
from typing import Any, Dict, Final, Optional

from napari.utils.events import EventEmitter

KEYBINDING_CONTEXT_ATTR = '_keybinding_context'


class Context(dict):
    """Just a dict that has a parent.

    If `__getitem__` or `get` can't find a key, they check the parent.
    use `collect` to get the merged dict (self takes precedence)
    """

    _parent: Optional[Context] = None
    _id: int

    def __missing__(self, name: str) -> Any:
        return (self._parent or {})[name]

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        return (self._parent or {}).get(key, default)

    def collect(self) -> dict:
        """Collect all values including those provided by parent(s)"""
        p = self._parent.collect() if self._parent is not None else {}
        return {**p, **self}

    def _spawn(self, *a, **k) -> Context:
        ctx = Context(*a, **k)
        ctx._parent = self
        return ctx

    def __repr__(self) -> str:
        myvals = dict(self)
        parvals = {k: v for k, v in self.collect().items() if k not in self}
        parstring = f' # <{parvals}>' if parvals else ''
        return f'{type(self).__name__}({myvals}){parstring}'


class SettingsAwareContext(Context):
    """A special context that allows access of settings using `_key_prefix`."""

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
            val = self._settings
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


class _AbsContextKeyService(ABC):
    _parent: Optional[_AbsContextKeyService] = None

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

    def create_scoped(self, target: object = None):
        return ScopedContextKeyService(self, target)

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

    # vscode: createKey
    # one thing we don't have here is createKey, since we don't have
    # ContextKey objects.
    # outside of standard get/set methods the main thing this allows
    # is `reset()`... but this can be accomplished with a RawContextKey
    # and the descriptor protocol

    # vscode: "getContext"
    # used all over the place to retrieve the context for a given object
    # the context_attr is set on the object when creating a ScopedContext...
    def get_context(self, target: object) -> Context:
        """Return the context for an object (set with create_scoped)."""
        id = getattr(target, KEYBINDING_CONTEXT_ATTR, 0)
        return self._get_context_container(id)

    @property
    def _my_context(self) -> Context:
        """Return context for this service."""
        return self._get_context_container(self._context_id)

    # vscode: "getContextValuesContainer" (only used in this module)
    @abstractmethod
    def _get_context_container(self, context_id: int) -> Context:
        """Get Context dict for `context_id`"""

    # vscode: "disposeContext" (only used in this module)
    @abstractmethod
    def del_context(self, context_id: int) -> None:
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


class ContextKeyService(_AbsContextKeyService):
    """A root context."""

    _contexts: Dict[int, Context]
    __instance: Optional[ContextKeyService] = None

    def __init__(self) -> None:
        super().__init__(0)
        self._ctx_count: count[int] = count(1)
        my_ctx = SettingsAwareContext(self._context_id, self.context_changed)
        self._contexts = {self._context_id: my_ctx}

    def _get_context_container(self, context_id: int) -> Context:
        return self._contexts.get(context_id, Context())  # Null context

    def del_context(self, context_id: int) -> None:
        self._contexts.pop(context_id, None)

    def create_child_context(self, parent_ctx_id: Optional[int] = None) -> int:
        _id = next(self._ctx_count)
        pid = parent_ctx_id or self._context_id
        if pid in self._contexts:
            self._contexts[_id] = self._contexts[pid]._spawn()
        else:
            self._contexts[_id] = Context()
        return _id

    def __repr__(self):
        s = ''
        for item in self._contexts.values():
            if item is self._my_context:
                s += pprint.pformat(dict(self))
            else:
                s += f', {id(item)}'
        return f'{type(self).__name__}({s})'

    @classmethod
    def instance(cls) -> ContextKeyService:
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance


class ScopedContextKeyService(_AbsContextKeyService):
    """A child context of ContextKeyService, scoped on some object."""

    _parent: _AbsContextKeyService

    # TODO: figure out whether we need the scope object
    def __init__(self, parent: _AbsContextKeyService, obj: object) -> None:
        import weakref

        if hasattr(obj, KEYBINDING_CONTEXT_ATTR):
            c = getattr(obj, KEYBINDING_CONTEXT_ATTR)
            raise RuntimeError(f"{obj} is already scoped on context {c}")
        super().__init__(parent.create_child_context())
        self._parent = parent
        self._parent.context_changed.connect(self._reemit_event)

        if obj is not None:
            # when the object gets deleted, delete this scoped_context from root
            weakref.finalize(obj, self._parent.del_context, self._context_id)
            # save weakref to object, so we can cleanup the context_attr if
            # this scoped context gets deleted.
            self._obj_ref = weakref.ref(obj)
            with contextlib.suppress(AttributeError):
                setattr(obj, KEYBINDING_CONTEXT_ATTR, self._context_id)

    def __del__(self):
        """Delete this scoped service, cleanup parent, remove attr"""
        if self._parent is not None:
            self._parent.context_changed.disconnect(self._reemit_event)
            self._parent.del_context(self._context_id)
        with contextlib.suppress(AttributeError):
            delattr(self._obj_ref(), KEYBINDING_CONTEXT_ATTR)

    # aka: getContextValuesContainer
    def _get_context_container(self, context_id: int) -> Context:
        return self._parent._get_context_container(context_id)

    def del_context(self, context_id: int) -> None:
        self._parent.del_context(context_id)

    def create_child_context(self, parent_ctx_id: Optional[int] = None) -> int:
        return self._parent.create_child_context(parent_ctx_id)

    def _reemit_event(self, event):
        self.context_changed(value=event.value)

    def __repr__(self):
        _id = hex(id(self))
        return f'{type(self).__name__}(id={_id}, {pprint.pformat(dict(self))})'
