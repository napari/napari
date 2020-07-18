import dataclasses as _dataclasses
from typing import Any, Callable, Type, TypeVar

import toolz as tz

from .event import EmitterGroup

WHEN_SET = "_on_{name}_set"
T = TypeVar("T")


def make_post_init(
    orig_post_init: Callable[..., None] = None, events=False, properties=False,
) -> Callable[..., None]:
    def _event_post_init(self, *initvars) -> None:
        emitter_group = EmitterGroup(
            source=self,
            auto_connect=False,
            **{name: None for name in self.__dataclass_fields__},
        )
        object.__setattr__(self, 'events', emitter_group)
        if orig_post_init is not None:
            orig_post_init(self, *initvars)
        if properties:
            # This should happen after initialization to handle default
            # factories in dataclasses
            convert_fields_to_properties(self)
        object.__setattr__(self, '_is_initialized', True)

    return _event_post_init


def setattr_with_events(self: T, name: str, value: Any) -> None:
    object.__setattr__(self, name, value)
    if name in self.__annotations__ and getattr(
        self, '_is_initialized', False
    ):
        # if custom set method `_on_<name>_set` exists, call it
        setter_method = getattr(self, WHEN_SET.format(name=name), None)
        if callable(setter_method):
            # the method can return True, if it wants to handle its own events
            if setter_method(getattr(self, name)):
                return
        # otherwise, we emit this event
        if name in self.events:  # type: ignore # (needs SupportsEvents)
            getattr(self.events, name)(value=getattr(self, name))  # type: ignore


def make_getter(name):
    def fget(self):
        return getattr(self, name)

    return fget


def make_setter(name):
    def fset(self, value):
        setattr(self, name, value)

    return fset


def convert_fields_to_properties(obj):
    from numpydoc.docscrape import ClassDoc

    cls = obj.__class__
    cls_doc = ClassDoc(cls)
    params = {p.name: p for p in cls_doc["Parameters"]}
    for field in _dataclasses.fields(cls):

        private_name = f"_{field.name}"
        setattr(obj, private_name, getattr(obj, field.name))
        fget = make_getter(private_name)
        fset = make_setter(private_name)
        doc = None
        if field.name in params:
            param = params[field.name]
            doc = "\n".join(param.desc)
            # TODO: could compare param.type to field.type here for consistency

        prop = property(fget=fget, fset=fset, fdel=None, doc=doc)
        setattr(cls, field.name, prop)


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
    as class attributes with a type annotation.

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

    if properties and frozen:
        raise ValueError("`properties=True` cannot be used with `frozen=True`")
    if events or properties:
        orig_post_init = getattr(cls, '__post_init__', None)
        post_init = make_post_init(orig_post_init, events, properties)
        setattr(cls, '__post_init__', post_init)

    _cls = _dataclasses._process_class(  # type: ignore
        cls, init, repr, eq, order, unsafe_hash, frozen
    )

    if events:
        setattr(_cls, '_is_initialized', False)
        _cls.__setattr__ = setattr_with_events
    return _cls
