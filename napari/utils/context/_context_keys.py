from __future__ import annotations

from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
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

from ._expressions import Name

if TYPE_CHECKING:
    from napari.utils.events import Event, EventEmitter

    from ._service import _BaseContextKeyService


T = TypeVar("T")
A = TypeVar("A")


class __missing:
    ...


MISSING = __missing()


class RawContextKey(Name, Generic[A, T]):
    class Info(NamedTuple):
        key: str
        type: Optional[Type]
        description: Optional[str]

    _info: List[Info] = []

    def __init__(
        self,
        key: str,
        default_value: Union[T, __missing] = MISSING,
        description: Optional[str] = None,
        updater: Optional[Callable[[A], T]] = None,
        *,
        hide: bool = False,
    ) -> None:
        super().__init__(key)
        self._default_value = default_value
        self._updater = updater
        if not hide:
            type_ = (
                type(default_value)
                if default_value not in (None, MISSING)
                else None
            )
            self._info.append(RawContextKey.Info(key, type_, description))

    def __str__(self) -> str:
        return self.id

    @classmethod
    def info(cls) -> List[RawContextKey.Info]:
        return list(cls._info)

    def bind_to(self, service: _BaseContextKeyService) -> None:
        service[self.id] = self._default_value

    def __set_name__(self, owner: Type, name):
        if name != self.id:
            raise ValueError(
                "Please use the same name for the class attribute and the key:"
                f"\n{type(owner).__name__}.{name} != {self.key}"
            )

    @overload
    def __get__(
        self, obj: Literal[None], objtype: Type
    ) -> RawContextKey[A, T]:
        """When we got from the class, we return ourself"""

    @overload
    def __get__(self, obj: ContextNamespace, objtype: Type) -> T:
        """When we got from the object, we return the current value"""

    def __get__(
        self, obj: Optional[ContextNamespace], objtype=Type
    ) -> Union[T, None, RawContextKey[A, T]]:
        if obj is None:
            return self
        return obj._service[self.id]

    def __set__(self, obj: ContextNamespace, value: T) -> None:
        obj._service[self.id] = value

    def __delete__(self, obj: ContextNamespace) -> None:
        del obj._service[self.id]


class ContextNamespace:
    _service: _BaseContextKeyService
    _defaults: Dict[str, Any]
    _updaters: Dict[str, Callable]

    @classmethod
    def bind_to_service(
        cls, service: _BaseContextKeyService
    ) -> ContextNamespace:
        obj = cls()
        obj._service = service
        obj._defaults = {}
        obj._updaters = {}
        for k, v in type(obj).__dict__.items():
            if isinstance(v, RawContextKey):
                obj._defaults[k] = v._default_value
                v.bind_to(service)
                if callable(v._updater):
                    obj._updaters[k] = v._updater
        return obj

    def follow(self, on: EventEmitter, until: Optional[EventEmitter] = None):
        from napari.utils.events import Event

        on.connect(self._update)
        e = Event(type='null')
        e._push_source(on.source)
        self._update(e)
        if until is not None:
            until.connect(partial(on.disconnect, self._update))

    def _update(self, event: Event) -> None:
        for k, updater in self._updaters.items():
            setattr(self, k, updater(event.source))

    def reset(self, key: str) -> None:
        val = self._defaults[key]
        if val is MISSING:
            delattr(self, key)
        else:
            setattr(self, key, self._defaults[key])

    def reset_all(self) -> None:
        for key, default in self._defaults.items():
            setattr(self, key, default)

    def dict(self):
        return {k: getattr(self, k) for k in self._defaults}

    def __repr__(self):
        import pprint

        return pprint.pformat(self.dict())
