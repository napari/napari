from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

from psygnal import Signal

T = TypeVar("T")


class Context:
    def __init__(self, id: int, parent: Optional[Context] = None) -> None:
        self._id = id
        self._parent = parent
        self._value = {"_contextId": id}

    def set_value(self, key: str, value: Any) -> bool:
        if self._value.get(key) != value:
            self._value[key] = value
            return True
        return False

    def get_value(self, key: str) -> Any:
        val = self._value.get(key, "undefined")
        if val == "undefined" and self._parent:
            return self._parent.get_value(key)
        return None if val == "undefined" else val

    def remove_value(self, key: str) -> bool:
        if key in self._value:
            del self._value[key]
            return True
        return False

    def update_parent(self, parent: Context) -> None:
        self._parent = parent


class NullContext(Context):
    __instance: Optional[NullContext] = None

    def __init__(self) -> None:
        super().__init__(-1, None)

    @classmethod
    def instance(cls) -> NullContext:
        if cls.__instance is None:
            cls.__instance = NullContext()
        return cls.__instance


class ContextKey(Generic[T]):
    def __init__(
        self,
        service: AbstractContextKeyService,
        key: str,
        default_value: Optional[T],
    ) -> None:
        self._service = service
        self._key = key
        self._default_value = default_value
        self.reset()

    def get(self) -> Optional[T]:
        return self._service.get_context_key_value(self._key)

    def set(self, value: T) -> None:
        self._service.set_context(self._key, value)

    def reset(self) -> None:
        if self._default_value is None:
            self._service.remove_context(self._key)
        else:
            self._service.set_context(self._key, self._default_value)


class AbstractContextKeyService:
    contextChanged = Signal()

    def __init__(self, id: int) -> None:
        self._is_disposed: bool = False
        self._context_id = id

    # def change_events_buffered(self): ...

    @property
    def context_id(self) -> int:
        return self._context_id

    def create_key(
        self, key: str, default_value: Optional[T]
    ) -> ContextKey[T]:
        if self._is_disposed:
            raise RuntimeError(f"{type(self)} has been disposed.")
        return ContextKey(self, key, default_value)

    def set_context(self, key: str, value: Any) -> None:
        if self._is_disposed:
            return
        context = self.get_context_values_container(self._context_id)
        if not context:
            return
        if context.set_value(key, value):
            self.contextChanged.emit(key)

    def get_context_key_value(self, key: str) -> Any:
        if self._is_disposed:
            return None
        return self.get_context_values_container(self._context_id).get_value(
            key
        )

    def remove_context(self, key: str) -> None:
        if self._is_disposed:
            return
        if self.get_context_values_container(self._context_id).remove_value(
            key
        ):
            self.contextChanged.emit(key)

    @abstractmethod
    def dispose(self) -> None:
        ...

    @abstractmethod
    def get_context_values_container(self, context_id: int) -> Context:
        ...


class ContextKeyService(AbstractContextKeyService):
    _contexts: Dict[int, Context]

    def __init__(self) -> None:
        super().__init__(0)
        self._to_dispose = set()
        self._last_context_id = 0
        self._contexts = {self._context_id: Context(self._context_id)}

    def dispose(self) -> None:
        self.contextChanged._slots.clear()
        self._is_disposed = True
        for item in self._to_dispose:
            item.dispose()

    def get_context_values_container(self, context_id) -> Context:
        if not self._is_disposed and (context_id in self._contexts):
            return self._contexts[context_id]
        return NullContext.instance()

    def create_child_context(self, parent_id: Optional[int] = None) -> int:
        parent_id = parent_id or self._context_id
        if self._is_disposed:
            raise RuntimeError(f"{type(self)} has been disposed.")
        self._last_context_id += 1
        id = self._last_context_id
        self._contexts[id] = Context(
            id, self.get_context_values_container(parent_id)
        )
        return id
