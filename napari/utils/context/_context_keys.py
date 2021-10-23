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
    from ...utils.events import Event, EventEmitter
    from ._context import Context


T = TypeVar("T")
A = TypeVar("A")


class __missing:
    ...


MISSING = __missing()


class ContextKeyInfo(NamedTuple):
    key: str
    type: Optional[Type]
    description: Optional[str]


class RawContextKey(Name, Generic[A, T]):

    _info: List[ContextKeyInfo] = []

    def __init__(
        self,
        default_value: Union[T, __missing] = MISSING,
        description: Optional[str] = None,
        updater: Optional[Callable[[A], T]] = None,
        *,
        id: str = '',  # optional because of __set_name__
        hide: bool = False,
    ) -> None:
        super().__init__(id or '')
        self._default_value = default_value
        self._updater = updater
        self._description = description
        self._type = (
            type(default_value)
            if default_value not in (None, MISSING)
            else None
        )
        self._hidden = hide
        if id and not hide:
            self._store()

    def __str__(self) -> str:
        return self.id

    @classmethod
    def info(cls) -> List[ContextKeyInfo]:
        return list(cls._info)

    def _store(self) -> None:
        self._info.append(
            ContextKeyInfo(self.id, self._type, self._description)
        )

    def __set_name__(self, owner: Type, name: str):
        if self.id:
            raise ValueError(
                f"Cannot change id of RawContextKey (already {self.id!r})"
            )
        self.id = name
        if not self._hidden:
            self._store()

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
        return obj._context[self.id]

    def __set__(self, obj: ContextNamespace, value: T) -> None:
        obj._context[self.id] = value

    def __delete__(self, obj: ContextNamespace) -> None:
        del obj._context[self.id]


class ContextNamespace:
    def __init__(self, context: Context) -> None:
        self._context = context
        self._defaults: Dict[str, Any] = {}
        self._updaters: Dict[str, Callable] = {}
        for k, v in type(self).__dict__.items():
            if isinstance(v, RawContextKey):
                self._defaults[k] = context[v.id] = v._default_value
                if callable(v._updater):
                    self._updaters[k] = v._updater

    def follow(self, on: EventEmitter, until: Optional[EventEmitter] = None):
        from ...utils.events import Event

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
