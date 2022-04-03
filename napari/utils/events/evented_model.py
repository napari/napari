import operator
import sys
import warnings
from contextlib import contextmanager
from typing import Any, Callable, ClassVar, Dict, Set, Union

import numpy as np
from pydantic import BaseModel, PrivateAttr, main, utils

from ...utils.misc import pick_equality_operator
from ..translations import trans
from .event import EmitterGroup, Event

# encoders for non-napari specific field types.  To declare a custom encoder
# for a napari type, add a `_json_encode` method to the class itself.
# it will be added to the model json_encoders in :func:`EventedMetaclass.__new__`
_BASE_JSON_ENCODERS = {np.ndarray: lambda arr: arr.tolist()}


@contextmanager
def no_class_attributes():
    """Context in which pydantic.main.ClassAttribute just passes value 2.

    Due to a very annoying decision by PySide2, all class ``__signature__``
    attributes may only be assigned **once**.  (This seems to be regardless of
    whether the class has anything to do with PySide2 or not).  Furthermore,
    the PySide2 ``__signature__`` attribute seems to break the python
    descriptor protocol, which means that class attributes that have a
    ``__get__`` method will not be able to successfully retrieve their value
    (instead, the descriptor object itself will be accessed).

    This plays terribly with Pydantic, which assigns a ``ClassAttribute``
    object to the value of ``cls.__signature__`` in ``ModelMetaclass.__new__``
    in order to avoid masking the call signature of object instances that have
    a ``__call__`` method (https://github.com/samuelcolvin/pydantic/pull/1466).

    So, because we only get to set the ``__signature__`` once, this context
    manager basically "opts-out" of pydantic's ``ClassAttribute`` strategy,
    thereby directly setting the ``cls.__signature__`` to an instance of
    ``inspect.Signature``.

    For additional context, see:
    - https://github.com/napari/napari/issues/2264
    - https://github.com/napari/napari/pull/2265
    - https://bugreports.qt.io/browse/PYSIDE-1004
    - https://codereview.qt-project.org/c/pyside/pyside-setup/+/261411
    """

    if "PySide2" not in sys.modules:
        yield
        return

    # monkey patch the pydantic ClassAttribute object
    # the second argument to ClassAttribute is the inspect.Signature object
    def _return2(x, y):
        return y

    main.ClassAttribute = _return2
    try:
        yield
    finally:
        # undo our monkey patch
        main.ClassAttribute = utils.ClassAttribute


class EventedMetaclass(main.ModelMetaclass):
    """pydantic ModelMetaclass that preps "equality checking" operations.

    A metaclass is the thing that "constructs" a class, and ``ModelMetaclass``
    is where pydantic puts a lot of it's type introspection and ``ModelField``
    creation logic.  Here, we simply tack on one more function, that builds a
    ``cls.__eq_operators__`` dict which is mapping of field name to a function
    that can be called to check equality of the value of that field with some
    other object.  (used in ``EventedModel.__eq__``)

    This happens only once, when an ``EventedModel`` class is created (and not
    when each instance of an ``EventedModel`` is instantiated).
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        with no_class_attributes():
            cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        cls.__eq_operators__ = {}
        for n, f in cls.__fields__.items():
            cls.__eq_operators__[n] = pick_equality_operator(f.type_)
            # If a field type has a _json_encode method, add it to the json
            # encoders for this model.
            # NOTE: a _json_encode field must return an object that can be
            # passed to json.dumps ... but it needn't return a string.
            if hasattr(f.type_, '_json_encode'):
                encoder = f.type_._json_encode
                cls.__config__.json_encoders[f.type_] = encoder
                # also add it to the base config
                # required for pydantic>=1.8.0 due to:
                # https://github.com/samuelcolvin/pydantic/pull/2064
                EventedModel.__config__.json_encoders[f.type_] = encoder
        # check for @_.setters defined on the class, so we can allow them
        # in EventedModel.__setattr__
        cls.__property_setters__ = {}
        for name, attr in namespace.items():
            if isinstance(attr, property) and attr.fset is not None:
                cls.__property_setters__[name] = attr
        cls.__field_dependents__ = _get_field_dependents(cls)
        return cls


def _get_field_dependents(cls: 'EventedModel') -> Dict[str, Set[str]]:
    """Return mapping of field name -> dependent set of property names.

    Dependencies may be declared in the Model Config to emit an event
    for a computed property when a model field that it depends on changes
    e.g.  (@property 'c' depends on model fields 'a' and 'b')

    Examples
    --------
        class MyModel(EventedModel):
            a: int = 1
            b: int = 1

            @property
            def c(self) -> List[int]:
                return [self.a, self.b]

            @c.setter
            def c(self, val: Sequence[int]):
                self.a, self.b = val

            class Config:
                dependencies={'c': ['a', 'b']}
    """
    if not cls.__property_setters__:
        return {}

    deps: Dict[str, Set[str]] = {}

    _deps = getattr(cls.__config__, 'dependencies', None)
    if _deps:
        for prop, fields in _deps.items():
            if prop not in cls.__property_setters__:
                raise ValueError(
                    'Fields with dependencies must be property.setters. '
                    f'{prop!r} is not.'
                )
            for field in fields:
                if field not in cls.__fields__:
                    warnings.warn(f"Unrecognized field dependency: {field}")
                deps.setdefault(field, set()).add(prop)
    else:
        # if dependencies haven't been explicitly defined, we can glean
        # them from the property.fget code object:
        for prop, setter in cls.__property_setters__.items():
            for name in setter.fget.__code__.co_names:
                if name in cls.__fields__:
                    deps.setdefault(name, set()).add(prop)
    return deps


class EventedModel(BaseModel, metaclass=EventedMetaclass):
    """A Model subclass that emits an event whenever a field value is changed.

    Note: As per the standard pydantic behavior, default Field values are
    not validated (#4138) and should be correctly typed.
    """

    # add private attributes for event emission
    _events: EmitterGroup = PrivateAttr(default_factory=EmitterGroup)

    # mapping of name -> property obj for methods that are property setters
    __property_setters__: ClassVar[Dict[str, property]]
    # mapping of field name -> dependent set of property names
    # when field is changed, an event for dependent properties will be emitted.
    __field_dependents__: ClassVar[Dict[str, Set[str]]]
    __eq_operators__: ClassVar[Dict[str, Callable[[Any, Any], bool]]]
    __slots__: ClassVar[Set[str]] = {"__weakref__"}  # type: ignore

    # pydantic BaseModel configuration.  see:
    # https://pydantic-docs.helpmanual.io/usage/model_config/
    class Config:
        # whether to allow arbitrary user types for fields (they are validated
        # simply by checking if the value is an instance of the type). If
        # False, RuntimeError will be raised on model declaration
        arbitrary_types_allowed = True
        # whether to perform validation on assignment to attributes
        validate_assignment = True
        # whether to treat any underscore non-class var attrs as private
        # https://pydantic-docs.helpmanual.io/usage/models/#private-model-attributes
        underscore_attrs_are_private = True
        # whether to validate field defaults (default: False)
        # see https://github.com/napari/napari/pull/4138 before changing.
        validate_all = False
        # https://pydantic-docs.helpmanual.io/usage/exporting_models/#modeljson
        # NOTE: json_encoders are also added EventedMetaclass.__new__ if the
        # field declares a _json_encode method.
        json_encoders = _BASE_JSON_ENCODERS

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._events.source = self
        # add event emitters for each field which is mutable
        event_names = [
            name
            for name, field in self.__fields__.items()
            if field.field_info.allow_mutation
        ]
        event_names.extend(self.__property_setters__)
        self._events.add(**dict.fromkeys(event_names))

    def _super_setattr_(self, name: str, value: Any) -> None:
        # pydantic will raise a ValueError if extra fields are not allowed
        # so we first check to see if this field has a property.setter.
        # if so, we use it instead.
        if name in self.__property_setters__:
            self.__property_setters__[name].fset(self, value)
        else:
            super().__setattr__(name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in getattr(self, 'events', {}):
            # fallback to default behavior
            self._super_setattr_(name, value)
            return

        # grab current value
        before = getattr(self, name, object())

        # set value using original setter
        self._super_setattr_(name, value)

        # if different we emit the event with new value
        after = getattr(self, name)
        are_equal = self.__eq_operators__.get(name, operator.eq)
        if not are_equal(after, before):
            getattr(self.events, name)(value=after)  # emit event

            # emit events for any dependent computed property setters as well
            for dep in self.__field_dependents__.get(name, {}):
                getattr(self.events, dep)(value=getattr(self, dep))

    # expose the private EmitterGroup publically
    @property
    def events(self) -> EmitterGroup:
        return self._events

    @property
    def _defaults(self):
        return get_defaults(self)

    def reset(self):
        """Reset the state of the model to default values."""
        for name, value in self._defaults.items():
            if isinstance(value, EventedModel):
                getattr(self, name).reset()
            elif (
                self.__config__.allow_mutation
                and self.__fields__[name].field_info.allow_mutation
            ):
                setattr(self, name, value)

    def update(
        self, values: Union['EventedModel', dict], recurse: bool = True
    ) -> None:
        """Update a model in place.

        Parameters
        ----------
        values : dict, napari.utils.events.EventedModel
            Values to update the model with. If an EventedModel is passed it is
            first converted to a dictionary. The keys of this dictionary must
            be found as attributes on the current model.
        recurse : bool
            If True, recursively update fields that are EventedModels.
            Otherwise, just update the immediate fields of this EventedModel,
            which is useful when the declared field type (e.g. ``Union``) can have
            different realized types with different fields.
        """
        if isinstance(values, self.__class__):
            values = values.dict()
        if not isinstance(values, dict):
            raise ValueError(
                trans._(
                    "Unsupported update from {values}",
                    deferred=True,
                    values=type(values),
                )
            )

        with self.events.blocker() as block:
            for key, value in values.items():
                field = getattr(self, key)
                if isinstance(field, EventedModel) and recurse:
                    field.update(value, recurse=recurse)
                else:
                    setattr(self, key, value)

        if block.count:
            self.events(Event(self))

    def __eq__(self, other) -> bool:
        """Check equality with another object.

        We override the pydantic approach (which just checks
        ``self.dict() == other.dict()``) to accommodate more complicated types
        like arrays, whose truth value is often ambiguous. ``__eq_operators__``
        is constructed in ``EqualityMetaclass.__new__``
        """
        if not isinstance(other, EventedModel):
            return self.dict() == other

        for f_name, eq in self.__eq_operators__.items():
            if f_name not in other.__eq_operators__:
                return False
            if (
                hasattr(self, f_name)
                and hasattr(other, f_name)
                and not eq(getattr(self, f_name), getattr(other, f_name))
            ):
                return False
        return True

    @contextmanager
    def enums_as_values(self, as_values: bool = True):
        """Temporarily override how enums are retrieved.

        Parameters
        ----------
        as_values : bool, optional
            Whether enums should be shown as values (or as enum objects),
            by default `True`
        """
        null = object()
        before = getattr(self.Config, 'use_enum_values', null)
        self.Config.use_enum_values = as_values
        try:
            yield
        finally:
            if before is not null:
                self.Config.use_enum_values = before
            else:
                delattr(self.Config, 'use_enum_values')


def get_defaults(obj: BaseModel):
    """Get possibly nested default values for a Model object."""
    dflt = {}
    for k, v in obj.__fields__.items():
        d = v.get_default()
        if d is None and isinstance(v.type_, main.ModelMetaclass):
            d = get_defaults(v.type_)
        dflt[k] = d
    return dflt
