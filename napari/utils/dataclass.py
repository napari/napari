import dataclasses as _pydcls
import typing
from typing import Any, Callable, ClassVar, Type, TypeVar, Set

import toolz as tz

from .event import EmitterGroup

WHEN_SET = "_on_{name}_set"
T = TypeVar("T")


@tz.curry
def set_with_events(self: T, name: str, value: Any, fields: Set[str]) -> None:
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
    self : T
        An instance of the decorated dataclass of Type[T]
    name : str
        The name of the attribute being set.
    value : Any
        The new value for the attribute.
    fields : set of str
        Only emit events for field names in this set.
    """
    # first call the original
    object.__setattr__(self, name, value)
    if name in fields:
        # if custom set method `_on_<name>_set` exists, call it
        setter_method = getattr(self, WHEN_SET.format(name=name), None)
        if callable(setter_method):
            # the method can return True, if it wants to handle its own events
            if setter_method(getattr(self, name)):
                return
        # otherwise, we emit the event
        if name in self.events:  # type: ignore # (needs SupportsEvents)
            # use gettattr again in case `_on_name_set` has modified it
            getattr(self.events, name)(value=getattr(self, name))  # type: ignore


def add_events_to_class(cls: Type[T]) -> Callable[..., None]:
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

    def evented_post_init(self: T, *initvars) -> None:
        # create an EmitterGroup with an EventEmitter for each field
        # in the dataclass
        emitter_group = EmitterGroup(
            source=self,
            auto_connect=False,
            **{n: None for n in getattr(self, '__dataclass_fields__', {})},
        )
        object.__setattr__(self, 'events', emitter_group)
        # call original __post_init__
        if orig_post_init is not None:
            orig_post_init(self, *initvars)
        # modify __setattr__ with version that emits an event when setting
        setter = set_with_events(fields={f.name for f in _pydcls.fields(cls)})
        setattr(cls, '__setattr__', setter)

    setattr(cls, '__post_init__', evented_post_init)


def convert_fields_to_properties(cls: Type[T]):
    """Convert all fields in a dataclass instance to property descriptors.

    Note: this modifies class Type[T] (the class that was decorated with
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
        An instance of class ``Type[T]`` that has been deorated as a dataclass.
    """
    from numpydoc.docscrape import ClassDoc

    # grab docstring to populate properties docs
    cls_doc = ClassDoc(cls)
    params = {p.name: p for p in cls_doc["Parameters"]}

    # loop through annotated members of the glass
    for name, type_ in list(cls.__dict__.get('__annotations__', {}).items()):
        # ClassVar types are exempt from dataclasses and @properties
        # https://docs.python.org/3/library/dataclasses.html#class-variables
        if _pydcls._is_classvar(type_, typing):
            continue
        private_name = f"_{name}"
        # store the original value for the property
        default = getattr(cls, name, _pydcls.MISSING)

        # add the private_name as a ClassVar annotation on the original class
        # (annotations of type ClassVar are ignored by the dataclass)
        cls.__annotations__[private_name] = ClassVar[type_]

        # make the actual getter/setter functions that the property will use
        def fget(self, key=private_name):
            return getattr(self, key)

        def fset(self, value, key=private_name, default=default):
            # during __init__, the dataclass will try to set the instance
            # attribute to the property itself!  So we intervene and set it
            # to the default value from the dataclass declaration
            if type(value) is property:
                value = default
                # dataclasses may define attributes as field(...)
                # so we need to get the default value from the field
                if isinstance(default, _pydcls.Field):
                    # there will only ever be default_factory OR default
                    # otherwise an exception will have been raised earlier.
                    if default.default_factory is not _pydcls.MISSING:
                        value = default.default_factory()
                    elif default.default is not _pydcls.MISSING:
                        value = default.default
                # If the value is still missing, then it means that the user
                # failed to provide a required positional argument when
                # instantiating the dataclass.
                if value is _pydcls.MISSING:
                    _name = key.lstrip("_")
                    raise TypeError(
                        "__init__() missing required "
                        f"positional argument: '{_name}'"
                    )
            setattr(self, key, value)

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


@tz.curry
def dataclass(
    cls: Type[T],
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    events: bool = False,
    properties: bool = False,
) -> Type[T]:
    """Enhanced dataclass decorator with events and property descriptors.

    Examines PEP 526 __annotations__ to determine fields.  Fields are defined
    as class attributes with a type annotation.  Everything but ``events`` and
    ``properties`` are defined on the builtin dataclass decorator.

    Parameters
    ----------
    cls : Type[T]
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
        altered, by default False
    properties : bool, optional
        If true, field attributes will be converted to property descriptors.
        If the class has a class docstring in numpydocs format, docs for each
        property will be taken from the ``Parameters`` section for the
        corresponding parameter, by default False

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

    if properties:
        if frozen:
            raise ValueError(
                "`properties=True` cannot be used with `frozen=True`"
            )
        # convert public dataclass fields to properties
        convert_fields_to_properties(cls)

    if events:
        # create a modified __post_init__ method that creates an EmitterGroup
        add_events_to_class(cls)

    # if neither events or properties are True, this function is exactly like
    # the builtin `dataclasses.dataclass`
    _cls = _pydcls._process_class(
        cls, init, repr, eq, order, unsafe_hash, frozen
    )
    return _cls
