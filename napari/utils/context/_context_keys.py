from __future__ import annotations

from functools import partial
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    NamedTuple,
    Optional,
    Type,
    TypeVar,
    Union,
    overload,
)

if TYPE_CHECKING:
    from napari.utils.events import Event, EventEmitter

    from ._service import ContextKeyService


class ContextKeyInfo(NamedTuple):
    key: str
    type: Optional[Type]
    description: Optional[str]


T = TypeVar("T")
A = TypeVar("A")


class RawContextKey(Generic[A, T]):
    _info: List[ContextKeyInfo] = []

    def __init__(
        self,
        key: str,
        default_value: Optional[T] = None,
        description: Optional[str] = None,
        updater: Optional[Callable[[A], T]] = None,
        *,
        hide: bool = False,
    ) -> None:
        self.key = key
        self._default_value = default_value
        self._updater = updater
        if not hide:
            type_ = type(default_value) if default_value is not None else None
            self._info.append(ContextKeyInfo(key, type_, description))

    def __str__(self) -> str:
        return self.key

    @classmethod
    def all(cls) -> List[ContextKeyInfo]:
        return list(cls._info)

    def bind_to(self, service: ContextKeyService) -> None:
        service.create_key(self.key, self._default_value)

    def __set_name__(self, owner: Type, name):
        self.public_name = name

    @property
    def _private_name(self):
        return "_" + self.public_name

    @overload
    def __get__(
        self, obj: Literal[None], objtype: Type
    ) -> RawContextKey[A, T]:
        ...

    @overload
    def __get__(self, obj: CtxKeys, objtype: Type) -> T:
        ...

    def __get__(
        self, obj: Optional[CtxKeys], objtype=Type
    ) -> Union[T, None, RawContextKey[A, T]]:
        if obj is None:
            return self
        return obj._service.get_context_key_value(self.key)

    def __set__(self, obj: CtxKeys, value: T) -> None:
        obj._service.set_context(self.key, value)


class CtxKeys:
    def __init__(self, service: ContextKeyService) -> None:
        self._service = service
        # srv.create_key("LayerListId", f"LayerList:{id(layer_list)}")

        self._key_names = set()
        self._updaters: Dict[str, Callable] = {}
        for k, v in type(self).__dict__.items():
            if isinstance(v, RawContextKey):
                self._key_names.add(k)
                v.bind_to(service)
                if callable(v._updater):
                    self._updaters[k] = v._updater

    def follow(self, on: EventEmitter, until: Optional[EventEmitter] = None):
        on.connect(self._update)
        e = Event(type='null')
        e._push_source(on.source)
        self._update(e)
        if until is not None:
            until.connect(partial(on.disconnect, self._update))

    def _update(self, event: Event):
        for k, updater in self._updaters.items():
            setattr(self, k, updater(event.source))

    def dict(self):
        return {k: getattr(self, k) for k in self._key_names}

    def __repr__(self):
        import pprint

        return pprint.pformat(self.dict())
