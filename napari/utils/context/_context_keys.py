from __future__ import annotations

from functools import partial
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    MutableMapping,
    NamedTuple,
    Optional,
    Type,
    TypeVar,
    Union,
    overload,
)

from typing_extensions import Literal

from ..translations import trans
from ._expressions import Name

if TYPE_CHECKING:
    from ...utils.events import Event, EventEmitter


T = TypeVar("T")
A = TypeVar("A")


class __missing:
    """Sentinel... done this way for the purpose of typing."""

    def __repr__(self):
        return 'MISSING'


MISSING = __missing()


class ContextKeyInfo(NamedTuple):
    """Just a recordkeeping tuple.

    Retrieve all declared ContextKeys with ContextKeyInfo.info().
    """

    key: str
    type: Optional[Type]
    description: Optional[str]
    namespace: Optional[Type[ContextNamespace]]


class ContextKey(Name, Generic[A, T]):
    """Context key name, default, description, and getter.

    This is intended to be used as class attribute in a `ContextNamespace`.
    This is a subclass of `Name`, and is therefore usable in an `Expression`.
    (see examples.)

    Parameters
    ----------
    default_value : Any, optional
        The default value for this key, by default MISSING
    description : str, optional
        Description of this key.  Useful for documentation, by default None
    getter : callable, optional
        Callable that receives an object and retrieves the current value for
        this key, by default None.
        For example, if this ContextKey represented the length of some list,
        (like the layerlist) it might look like
        `length = ContextKey(0, 'length of the list', lambda x: len(x))`
    id : str, optional
        Explicitly provide the `Name` string used when evaluating a context,
        by default the key will be taken as the attribute name to which this
        object is assigned as a class attribute:

    Examples
    --------
    >>> class MyNames(ContextNamespace):
    ...     some_key = ContextKey(0, 'some description', lambda x: sum(x))

    >>> expr = MyNames.some_key > 5  # create an expression using this key

    these expressions can be later evaluated with some concrete context.

    >>> expr.eval({'some_key': 3})  # False
    >>> expr.eval({'some_key': 6})  # True
    """

    # This will catalog all ContextKeys that get instantiated, which provides
    # an easy way to organize documentation.
    # ContextKey.info() returns a list with info for all ContextKeys
    _info: List[ContextKeyInfo] = []

    def __init__(
        self,
        default_value: Union[T, __missing] = MISSING,
        description: Optional[str] = None,
        getter: Optional[Callable[[A], T]] = None,
        *,
        id: str = '',  # optional because of __set_name__
    ) -> None:
        super().__init__(id or '')
        self._default_value = default_value
        self._getter = getter
        self._description = description
        self._owner: Optional[Type[ContextNamespace]] = None
        self._type = (
            type(default_value)
            if default_value not in (None, MISSING)
            else None
        )
        if id:
            self._store()

    def __str__(self) -> str:
        return self.id

    @classmethod
    def info(cls) -> List[ContextKeyInfo]:
        return list(cls._info)

    def _store(self) -> None:
        self._info.append(
            ContextKeyInfo(self.id, self._type, self._description, self._owner)
        )

    def __set_name__(self, owner: Type[ContextNamespace[A]], name: str):
        """Set the name for this key.

        (this happens when you instantiate this class as a class attribute).
        """
        if self.id:
            raise ValueError(
                trans._(
                    "Cannot change id of ContextKey (already {identifier!r})",
                    deferred=True,
                    identifier=self.id,
                )
            )
        self._owner = owner
        self.id = name
        self._store()

    @overload
    def __get__(self, obj: Literal[None], objtype: Type) -> ContextKey[A, T]:
        """When we got from the class, we return ourself."""

    @overload
    def __get__(self, obj: ContextNamespace[A], objtype: Type) -> T:
        """When we got from the object, we return the current value."""

    def __get__(
        self, obj: Optional[ContextNamespace[A]], objtype=Type
    ) -> Union[T, None, ContextKey[A, T]]:
        """Get current value of the key in the associated context."""
        if obj is None:
            return self
        return obj._context.get(self.id, MISSING)

    def __set__(self, obj: ContextNamespace[A], value: T) -> None:
        """Set current value of the key in the associated context."""
        obj._context[self.id] = value

    def __delete__(self, obj: ContextNamespace[A]) -> None:
        """Delete key from the associated context."""
        del obj._context[self.id]


class ContextNamespaceMeta(type):
    """Metaclass that finds all ContextNamespace members."""

    def __new__(cls: Type, clsname: str, bases: tuple, attrs: dict):
        cls = super().__new__(cls, clsname, bases, attrs)
        cls._members_map_ = {
            k: v for k, v in attrs.items() if isinstance(v, ContextKey)
        }
        return cls

    @property
    def __members__(cls) -> MappingProxyType[str, ContextKey]:
        return MappingProxyType(cls._members_map_)

    def __dir__(self) -> List[str]:
        return [
            '__class__',
            '__doc__',
            '__members__',
            '__module__',
        ] + list(self._members_map_)


class ContextNamespace(Generic[A], metaclass=ContextNamespaceMeta):
    """A collection of related keys in a context

    meant to be subclassed, with `ContextKeys` as class attributes.
    """

    def __init__(self, context: MutableMapping) -> None:
        self._context = context

        # on instantiation we create an index of defaults and value-getters
        # to speed up retrieval later
        self._defaults: Dict[str, Any] = {}  # default values per key
        self._getters: Dict[str, Callable[[A], Any]] = {}  # value getters
        for name, ctxkey in type(self).__members__.items():
            self._defaults[name] = ctxkey._default_value
            if ctxkey._default_value is not MISSING:
                context[ctxkey.id] = ctxkey._default_value
            if callable(ctxkey._getter):
                self._getters[name] = ctxkey._getter

    def follow(self, on: EventEmitter, until: Optional[EventEmitter] = None):
        """Tell this context to update all keys when `on` is emitted.

        This will cause `self._update` to be called (updating *all* keys in the
        namespace) whenever `on` is emitted.  For finer tuned control (i.e.
        if you only want to update some of the keys), you may wish to create
        your own event bindings.

        Parameters
        ----------
        on : EventEmitter
            The event that should trigger an update of all of the keys in this
            namespace
        until : EventEmitter, optional
            An optional event that should stop the updating, by default None

        Examples
        --------
        >>> some_context = Context()
        >>> ctx_keys = ContextNamespace(some_context)
        >>> ctx_keys.follow(object.events.changed)
        # `ctx_keys` will update all of its associated keys in `some_context`
        # whenever object.events.changed is emitted
        """
        from ...utils.events import Event

        on.connect(self.update)

        # trigger the first update
        # we create a fake event to use the source from the `on` emitter
        e = Event(type='null')
        e._push_source(on.source)
        self.update(e)

        if until is not None:
            until.connect(partial(on.disconnect, self.update))

    def update(self, event: Event) -> None:
        """Trigger an update of all "getter" functions in this namespace."""
        for k, get in self._getters.items():
            setattr(self, k, get(event.source))

    def reset(self, key: str) -> None:
        """Reset keys to its default."""
        val = self._defaults[key]
        if val is MISSING:
            try:
                delattr(self, key)
            except KeyError:
                pass
        else:
            setattr(self, key, self._defaults[key])

    def reset_all(self) -> None:
        """Reset all keys to their defaults."""
        for key in self._defaults:
            self.reset(key)

    def dict(self):
        """Return all keys in this namespace."""
        return {k: getattr(self, k) for k in type(self).__members__}

    def __repr__(self):
        import pprint

        return pprint.pformat(self.dict())
