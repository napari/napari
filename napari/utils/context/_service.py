from __future__ import annotations

import pprint
from abc import ABC, abstractmethod
from contextlib import contextmanager
from itertools import count
from typing import Any, Dict, Iterable, Mapping, Optional, Union

from napari.utils.events import EventEmitter

Context = Dict[str, Any]
PARENT_KEY = '__parent__'


def _get_key_recurse_parents(dct: dict, key, _par_key=PARENT_KEY):
    if key in dct:
        return dct[key]
    if _par_key in dct:
        return _get_key_recurse_parents(dct[_par_key], key)


def _collect_all_values(dct: dict, _par_key=PARENT_KEY):
    result = _collect_all_values(dct[_par_key]) if _par_key in dct else {}
    result.update(dct)
    result.pop(_par_key, None)
    return result


class _AbsContextKeyService(ABC):
    def __init__(self, id: int) -> None:
        self._context_id = id
        self.context_changed = EventEmitter(self, 'context_changed')

    @property
    def context_id(self) -> int:
        return self._context_id

    @contextmanager
    def buffered_change_events(self):
        ...  # TODO

    def create_scoped(self, target=None):
        return ScopedContextKeyService(self, target)

    # TODO: rename get_context_key?
    # def get_context_key_value(self, key: str) -> Any:
    #     return _get_key_recurse_parents(self._my_context, key)

    def __getitem__(self, key: str) -> Any:
        return _get_key_recurse_parents(self._my_context, key)

    # TODO: rename set_context_key?
    # def set_context(self, key: str, value: Any) -> None:
    def __setitem__(self, key: str, value: Any) -> None:
        if self._my_context.get(key, '__missing__') != value:
            self._my_context[key] = value
            self.context_changed(value=[key])

    def update(
        self, _m: Union[Mapping, Iterable, None] = None, **kwargs
    ) -> None:
        d = dict(_m) if _m is not None else {}
        d.update(kwargs)
        for k, v in d.items():
            self.set_context(k, v)

    # TODO: rename del_context_key?
    # def remove_context(self, key: str) -> None:
    def __delitem__(self, key: str) -> None:
        if key in self._my_context:
            del self._my_context[key]
            self.context_changed(value=[key])

    @property
    def _my_context(self):
        return self.get_context(self._context_id)

    # aka: getContextValuesContainer
    @abstractmethod
    def get_context(self, id: int) -> Context:
        ...

    @abstractmethod
    def del_context(self, context_id: int) -> None:
        ...

    @abstractmethod
    def create_child_context(self, parent_ctx_id: Optional[int] = None) -> int:
        ...

    def dict(self):
        return _collect_all_values(self._my_context)

    def __iter__(self):
        yield from self.dict().items()

    def len(self):
        len(self.dict())


class ContextKeyService(_AbsContextKeyService):
    _contexts: Dict[int, Context]
    __instance: Optional[ContextKeyService] = None

    def __init__(self) -> None:
        super().__init__(0)
        self._ctx_count: count[int] = count(1)
        self._contexts = {self._context_id: {}}

    # aka: getContextValuesContainer
    def get_context(self, context_id: int) -> Context:
        return self._contexts.get(context_id, {})  # Null context

    def del_context(self, context_id: int) -> None:
        del self._contexts[context_id]

    def create_child_context(self, parent_ctx_id: Optional[int] = None) -> int:
        parent_id = parent_ctx_id or self._context_id
        ctx = {}
        if parent_id in self._contexts:
            ctx[PARENT_KEY] = self._contexts[parent_id]

        _id = next(self._ctx_count)
        self._contexts[_id] = ctx
        return _id

    def __repr__(self):
        return f'ContextKeyService({pprint.pformat(self._contexts)})'

    @classmethod
    def instance(cls) -> ContextKeyService:
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance


class ScopedContextKeyService(_AbsContextKeyService):
    # TODO: figure out whether we need the scope object
    def __init__(self, parent: _AbsContextKeyService, scope=None) -> None:
        super().__init__(parent.create_child_context())
        self._parent = parent
        self._updateParentChangeListener()
        self._scope = scope
        # TODO: set KEYBINDING_CONTEXT_ATTR on scope

    # aka: getContextValuesContainer
    def get_context(self, context_id: int) -> Context:
        return self._parent.get_context(context_id)

    def del_context(self, context_id: int) -> None:
        self._parent.del_context(context_id)

    def create_child_context(self, parent_ctx_id: Optional[int] = None) -> int:
        return self._parent.create_child_context(parent_ctx_id)

    def update_parent(self, new_parent: _AbsContextKeyService) -> None:
        current_ctx = self._my_context
        old_vals = _collect_all_values(current_ctx)
        self._parent.context_changed.disconnect(self._reemit)
        self._parent = new_parent
        self._updateParentChangeListener()
        new_ctx = self._parent.get_context(self._parent.context_id)
        current_ctx[PARENT_KEY] = new_ctx
        new_vals = _collect_all_values(current_ctx)
        changed_keys = [k for k, v in new_vals.items() if old_vals[k] != v]
        self.context_changed(value=changed_keys)

    def _updateParentChangeListener(self):
        self._parent.context_changed.connect(self._reemit)

    def _reemit(self, event):
        self.context_changed(value=event.value)

    def __repr__(self):
        return f'ScopedContextKeyService({pprint.pformat(self._my_context)})'
