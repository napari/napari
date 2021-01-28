import dataclasses as _dc
import operator
import typing
import warnings
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    NamedTuple,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

import dask.array as da
import numpy as np
import toolz as tz
from typing_extensions import (
    Annotated,
    _AnnotatedAlias,
    get_args,
    get_origin,
    get_type_hints,
)

try:
    from typing import _tp_cache
except ImportError:

    def _tp_cache(x):
        return x


from .event import EmitterGroup, Event
from .types import SupportsEvents

ON_SET = "_on_{name}_set"
ON_GET = "_on_{name}_get"
C = TypeVar("C")
T = TypeVar("T")
NO_ATTR = object()


class Property:
    """Declare a dataclass field as a property with getter/setter functions.

    This class is nearly an alias for `typing_extensions.Annotated` (or, in
    ≥python3.9, `typing.Annotated`), except that it provides stricter type
    checking.  See `__class_getitem__` for special considerations.

    ``Property`` should be used with the actual type, followed by an optional
    ``getter`` function (or ``None``), followed by an optional ``setter``
    function (or ``None``): e.g. ``Property[Blending, str, Blending]``

    Examples
    --------

    >>> @evented_dataclass
    ... class Test:
    ...     # x will be stored as an int but *retrieved* as a string.
    ...     x: Property[int, str, int]
    """

    def __new__(cls, *args, **kwargs):
        raise TypeError("Type Property cannot be instantiated")

    def __init_subclass__(cls, *args, **kwargs):
        raise TypeError(f"Cannot subclass {cls.__module__}.Property")

    @_tp_cache
    def __class_getitem__(cls, params):
        """Create the runtime type representation of a Property type.

        Unfortunately, in order for this to behave like an `Annotated` type,
        without us having to vendor a lot of the python
        typing library, this must return an instance of
        `typing_extensions._AnnotatedAlias` (which just points to
        `typing._AnnotatedAlias` after python 3.9).  Hence the private import.
        Since this works on python3.9 (when typing acquired the Annotated type)
        it seems unlikely to break in future python releases.  But it should be
        monitored.
        """
        if not isinstance(params, tuple) or not (1 < len(params) < 4):
            raise TypeError(
                "Property[...] should be used with exactly two or three "
                "arguments (a type, a getter, and an optional setter)"
            )
        msg = "Property[T, ...]: T must be a type."
        origin = typing._type_check(params[0], msg)
        if params[1] is not None and not callable(params[1]):
            raise TypeError(f"Property getter not callable: {params[1]}")
        if (
            len(params) > 2
            and params[2] is not None
            and not callable(params[2])
        ):
            raise TypeError(f"Property getter not callable: {params[1]}")
        metadata = tuple(params[1:])
        return _AnnotatedAlias(origin, metadata)


# from dataclasses.py in the standard library
def _is_classvar(a_type):
    # This test uses a typing internal class, but it's the best way to
    # test if this is a ClassVar.
    return a_type is typing.ClassVar or (
        type(a_type) is typing._GenericAlias
        and a_type.__origin__ is typing.ClassVar
    )


# from dataclasses.py in the standard library
def _is_initvar(a_type):
    # The module we're checking against is the module we're
    # currently in (dataclasses.py).
    return a_type is _dc.InitVar or type(a_type) is _dc.InitVar


@contextmanager
def stripped_annotated_types(cls):
    """Temporarily strip Annotated types (for cleaner function signatures)."""
    original_annotations = cls.__annotations__
    cls.__annotations__ = get_type_hints(cls, include_extras=False)
    yield
    cls.__annotations__ = original_annotations


def set_with_events(self: C, name: str, value: Any) -> None:
    """Modified __setattr__ method that emits an event when set.

    Events will *only* be emitted if the ``name`` of the attribute being set
    is one of the dataclass fields (i.e. ``name in self.__annotations__``),
    and the dataclass ``__post_init__` method has already been called.

    Also looks for and calls an optional ``_on_name_set()`` method afterwards.

    Order of operations:
        1. Call the original ``__setattr__`` function to set the value
        2. Look for an ``_on_name_set`` method on the object
            a. If present, call it with the current value
            b. That method can do anything (including changing the value, or
               emitting its own events if necessary).  If changing the value,
               it should check to make sure that it is different than the
               current value before setting, or a ``RecursionError`` may occur.
            c. If that method returns ``True``. Return *without* emitting
               an event.
        3. If ``_on_name_set`` has not returned ``True``, then emit an event
           from the EventEmitter with the corresponding ``name`` in the.
           e.g. ``self.events.<name>(value=value)``.

    Parameters
    ----------
    self : C
        An instance of the decorated dataclass of Type[C]
    name : str
        The name of the attribute being set.
    value : Any
        The new value for the attribute.
    fields : set of str
        Only emit events for field names in this set.
    """
    if name not in getattr(self, 'events', {}):
        # fallback to default behavior
        object.__setattr__(self, name, value)
        return

    # grab current value
    before = getattr(self, name, NO_ATTR)
    object.__setattr__(self, name, value)

    # if custom set method `_on_<name>_set` exists, call it
    setter_method = getattr(self, ON_SET.format(name=name), None)
    if callable(setter_method):
        # the method can return True, if it wants to handle its own events
        try:
            if setter_method(getattr(self, name)):
                return
        except Exception as e:
            if before is NO_ATTR:
                object.__delattr__(self, name)
            else:
                object.__setattr__(self, name, before)
            meth_name = f"{self.__class__.__name__}.{ON_SET.format(name=name)}"
            raise type(e)(f"Error in {meth_name} (value not set): {e}")

    # if different we emit the event with new value
    after = getattr(self, name)
    if not self.__equality_checks__.get(name, is_equal)(after, before):
        # use gettattr again in case `_on_name_set` has modified it
        getattr(self.events, name)(value=after)  # type: ignore


def is_equal(v1, v2):
    """
    Function for basic comparison compare.

    Parameters
    ----------
    v1: Any
        first value
    v2: Any
        second value

    Returns
    -------
    result : bool
        Returns ``True`` if ``v1`` and ``v2`` are equivalent.
        If an exception is raised during comparison, returns ``False``.

    Warns
    -----
    UserWarning
        If an exception is raised during comparison.
    """
    try:
        return bool(v1 == v2)
    except Exception as e:
        warnings.warn(
            "Comparison method failed. Returned False. "
            f"There may be need to define custom compare methods in __equality_checks__ dictionary. Exception {e}"
        )
        return False


def _type_to_compare(type_) -> Optional[Callable[[Any, Any], bool]]:
    """
    Try to determine compare function for types which cannot be compared with `operator.eq`.
    Currently support `numpy.ndarray` and `dask.core.Array`.
    Unpack `typing.Optional` and `dataclasses.InitVar` box.

    Parameters
    ----------
    type_: type
    type to examine

    Returns
    -------
    Optional[Callable[[Any, Any], bool]]
        Compare function
    """
    import inspect

    if isinstance(type_, _AnnotatedAlias):
        type_ = type_.__origin__
    if isinstance(type_, _dc.InitVar):
        type_ = type_.type

    # get main type from Optional[type]
    _args = get_args(type_)
    if (
        get_origin(type_) is Union
        and len(_args) == 2
        and isinstance(None, _args[1])
    ):
        type_ = _args[0]
    if not inspect.isclass(type_):
        if not getattr(type_, "__module__", None) == "typing":
            warnings.warn(f"Bug in type recognition {type_}")
        return
    if issubclass(type_, np.ndarray):
        return np.array_equal
    if issubclass(type_, da.core.Array):
        return operator.is_
    return None


def add_events_to_class(cls: Type[C]) -> Type[C]:
    """Return a new __post_init__ method wrapper with events.

    Parameters
    ----------
    cls : type
        The class being decorated as a dataclass

    Returns
    -------
    Callable[..., None]
        A modified __post_init__ method that wraps the original.
    """

    # get a handle to the original __post_init__ method if present
    orig_post_init: Callable[..., None] = getattr(cls, '__post_init__', None)

    _fields = [
        _dc._get_field(cls, name, type_)
        for name, type_ in cls.__dict__.get('__annotations__', {}).items()
    ]
    e_fields = {
        fld.name: None
        for fld in _fields
        if fld._field_type is _dc._FIELD and fld.metadata.get("events", True)
    }

    # create dict with compare functions for fields which cannot be compared
    # using standard equal operator, like numpy arrays.
    # it will be set to __equality_checks__ class parameter.
    compare_dict_base = getattr(cls, "__equality_checks__", {})
    compare_dict = {
        n: t
        for n, t in {
            name: _type_to_compare(type_)
            for name, type_ in cls.__dict__.get('__annotations__', {}).items()
            if name not in compare_dict_base
        }.items()
        if t is not None  # walrus operator is supported from python 3.8
    }
    # use compare functions provided by class creator.
    compare_dict.update(compare_dict_base)

    def evented_post_init(self: T, *initvars) -> None:
        # create an EmitterGroup with an EventEmitter for each field
        # in the dataclass, skip those with metadata={'events' = False}
        if hasattr(self, 'events') and isinstance(self.events, EmitterGroup):
            for em in self.events.emitters:
                e_fields.pop(em, None)
            self.events.add(**e_fields)
        else:
            self.events = EmitterGroup(
                source=self,
                auto_connect=False,
                **e_fields,
            )
        # call original __post_init__
        if orig_post_init is not None:
            orig_post_init(self, *initvars)

    # modify __setattr__ with version that emits an event when setting
    setattr(cls, '__setattr__', set_with_events)
    setattr(cls, '__post_init__', evented_post_init)
    setattr(cls, '__equality_checks__', compare_dict)
    return cls


# make the actual getter/setter functions that the property will use
@tz.curry
def prop_getter(priv_name: str, fcoerce: Callable, obj) -> Any:
    # val = object.__getattribute__(obj, name)
    value = fcoerce(getattr(obj, priv_name))
    pub_name = priv_name.lstrip("_")
    getter_method = getattr(obj, ON_GET.format(name=pub_name), None)
    if callable(getter_method):
        return getter_method(value)
    return value


@tz.curry
def prop_setter(
    priv_name: str, default: Any, fcoerce: Callable, obj, value
) -> None:
    # during __init__, the dataclass will try to set the instance
    # attribute to the property itself!  So we intervene and set it
    # to the default value from the dataclass declaration
    pub_name = priv_name.lstrip("_")
    if type(value) is property:
        value = default
        # dataclasses may define attributes as field(...)
        # so we need to get the default value from the field
        if isinstance(default, _dc.Field):
            default = cast(_dc.Field, default)
            # there will only ever be default_factory OR default
            # otherwise an exception will have been raised earlier.
            if default.default_factory is not _dc.MISSING:
                value = default.default_factory()
            elif default.default is not _dc.MISSING:
                value = default.default
        # If the value is still missing, then it means that the user
        # failed to provide a required positional argument when
        # instantiating the dataclass.
        if value is _dc.MISSING:
            raise TypeError(
                "__init__() missing required "
                f"positional argument: '{pub_name}'"
            )
    setattr(obj, priv_name, fcoerce(value))


class TypeGetSet(NamedTuple):
    '''A simple named tuple with a type, getter func, and setter func.'''

    type: Type[T]
    fget: Callable[[T], Any]
    fset: Callable[[Any], T]


@tz.curry
def _try_coerce(func, name, value):
    if func is not None:
        try:
            return func(value)
        except Exception as e:
            raise TypeError(f"Failed to coerce value {value} in {name}: {e}")
    return value


def parse_annotated_types(
    cls: Type,
) -> Dict[str, TypeGetSet]:
    """Return a dict of field names and their types.

    If any type annotations are instances of ``Property``, then their
    paramereters will be used to create type-coercion functions that will be
    called during getting/setting of this field.

    Parameters
    ----------
    cls : Type
        The class being processed as a dataclass

    Returns
    -------
    dict
        A dict of field names to ``TypeGetSet`` objects
    """
    out: Dict[str, TypeGetSet] = {}
    for name, typ in get_type_hints(cls, include_extras=True).items():
        d = [typ, None, None]
        if get_origin(typ) is Annotated:
            args = get_args(typ)
            d[: len(args)] = args
        d[1] = _try_coerce(d[1], name)
        d[2] = _try_coerce(d[2], name)
        out[name] = TypeGetSet(*d)
    return out


def convert_fields_to_properties(cls: Type[C]) -> Type[C]:
    """Convert all fields in a dataclass instance to property descriptors.

    Note: this modifies class Type[C] (the class that was decorated with
    ``@dataclass``) *after* instantiation of the class.  In other words, for a
    given field `f` on class `C`, `C.f` will *not* be a property descriptor
    until *after* C has been instantiated: `c = C()`.  (And reminder: property
    descriptors are class attributes).

    The reason for this is that dataclasses can have "default factory"
    functions that create default values for fields only during instantiation,
    and we don't want to have to recreate that logic here, (but we do need to
    know what the value of the field is).

    Parameters
    ----------
    obj : T
        An instance of class ``Type[C]`` that has been deorated as a dataclass.
    """
    from numpydoc.docscrape import ClassDoc

    # grab docstring to populate properties docs
    cls_doc = ClassDoc(cls)
    params = {p.name: p for p in cls_doc["Parameters"]}

    coerce_funcs = parse_annotated_types(cls)
    # loop through annotated members of the glass
    for name, type_ in list(cls.__dict__.get('__annotations__', {}).items()):
        # ClassVar and InitVar types are exempt from dataclasses and properties
        # https://docs.python.org/3/library/dataclasses.html#class-variables
        if _is_classvar(type_) or _is_initvar(type_):
            continue
        private_name = f"_{name}"
        # store the original value for the property
        default = getattr(cls, name, _dc.MISSING)

        # add the private_name as a ClassVar annotation on the original class
        # (annotations of type ClassVar are ignored by the dataclass)
        # `self.x` is the property, `self._x` contains the data
        cls.__annotations__[private_name] = ClassVar[type_]

        if name in coerce_funcs:
            fget = prop_getter(private_name, coerce_funcs[name].fget)
            fset = prop_setter(private_name, default, coerce_funcs[name].fset)
        else:
            fget = _try_coerce(None, name)
            fset = _try_coerce(None, name)
        # bring the docstring from the class to the property
        doc = None
        if name in params:
            param = params[name]
            doc = "\n".join(param.desc)
            # TODO: could compare param.type to field.type here for consistency
            # alternatively, we may just want to use pydantic for type
            # validation.

        # create the actual property descriptor
        prop = property(fget=fget, fset=fset, fdel=None, doc=doc)
        setattr(cls, name, prop)
    return cls


def update_from_dict(self, values, compare_fun=is_equal):
    if isinstance(values, self.__class__):
        values = values.asdict()
    if not isinstance(values, dict):
        raise ValueError(f"Unsupported update from {type(values)}")
    if all(compare_fun(values[k], v) for k, v in self.asdict().items()):
        return

    if isinstance(self, SupportsEvents):
        with self.events.blocker():
            for key, value in values.items():
                setattr(self, key, value)
        self.events(Event(self))
    else:
        for key, value in values.items():
            setattr(self, key, value)


@tz.curry
def evented_dataclass(
    cls: Type[C],
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    events: bool = True,
    properties: bool = True,
) -> Type[C]:
    """Enhanced dataclass decorator with events and property descriptors.

    Examines PEP 526 __annotations__ to determine fields.  Fields are defined
    as class attributes with a type annotation.  Everything but ``events`` and
    ``properties`` are defined on the builtin dataclass decorator.

    Note: if ``events==False`` and ``properties==False``, this is functionally
    equivalent to the builtin ``dataclasses.dataclass``

    Parameters
    ----------
    cls : Type[C]
        [description]
    init : bool, optional
        If  true, an __init__() method is added to the class, by default True
    repr : bool, optional
        If true, a __repr__() method is added, by default True
    eq : bool, optional
        [description], by default True
    order : bool, optional
        If true, rich comparison dunder methods are added, by default False
    unsafe_hash : bool, optional
        If true, a __hash__() method function is added, by default False
    frozen : bool, optional
        If true, fields may not be assigned to after instance creation, by
        default False
    events : bool, optional
        If true, an ``EmmitterGroup`` instance is added as attribute "events".
        Events will be emitted each time one of the dataclass fields are
        altered, by default True
    properties : bool, optional
        If true, field attributes will be converted to property descriptors.
        If the class has a class docstring in numpydocs format, docs for each
        property will be taken from the ``Parameters`` section for the
        corresponding parameter, by default True

    Returns
    -------
    decorated class
        Returns the same class as was passed in, with dunder methods
        added based on the fields defined in the class.

    Raises
    ------
    ValueError
        If both ``properties`` and ``frozen`` are True
    """

    if frozen and (events or properties):
        raise ValueError(
            "`frozen=True` is incompatible `properties=True` or `events=True`"
        )

    # TODO: currently, events must be process first here, otherwise
    # metada={'events':False} does not work... but that should be fixed.
    if events:
        # create a modified __post_init__ method that creates an EmitterGroup
        cls = add_events_to_class(cls)

    # if neither events or properties are True, this function is exactly like
    # the builtin `dataclasses.dataclass`
    with stripped_annotated_types(cls):
        cls = _dc.dataclass(  # type: ignore
            cls,
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
        )

    if properties:
        # convert public dataclass fields to properties
        cls = convert_fields_to_properties(cls)
    setattr(cls, 'asdict', _dc.asdict)
    setattr(cls, 'update', update_from_dict)
    return cls
