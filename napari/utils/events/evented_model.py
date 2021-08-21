import operator
import sys
import warnings
from contextlib import contextmanager
from typing import Any, Callable, ClassVar, Dict, Set

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
    main.ClassAttribute = lambda x, y: y
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
        return cls


class EventedModel(BaseModel, metaclass=EventedMetaclass):

    # add private attributes for event emission
    _events: EmitterGroup = PrivateAttr(default_factory=EmitterGroup)

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
        validate_all = True
        # https://pydantic-docs.helpmanual.io/usage/exporting_models/#modeljson
        # NOTE: json_encoders are also added EventedMetaclass.__new__ if the
        # field declares a _json_encode method.
        json_encoders = _BASE_JSON_ENCODERS

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._events.source = self
        # add event emitters for each field which is mutable
        fields = []
        for name, field in self.__fields__.items():
            if field.field_info.allow_mutation:
                fields.append(name)
        self._events.add(**dict.fromkeys(fields))

    def __setattr__(self, name, value):
        if name not in getattr(self, 'events', {}):
            # fallback to default behavior
            super().__setattr__(name, value)
            return

        # grab current value
        before = getattr(self, name, object())

        # set value using original setter
        super().__setattr__(name, value)

        # if different we emit the event with new value
        after = getattr(self, name)
        are_equal = self.__eq_operators__.get(name, operator.eq)
        if not are_equal(after, before):
            getattr(self.events, name)(value=after)  # emit event

    # expose the private EmitterGroup publically
    @property
    def events(self):
        return self._events

    @property
    def _defaults(self):
        return get_defaults(self)

    def reset(self):
        """Reset the state of the model to default values."""
        for name, value in self._defaults.items():
            if isinstance(value, EventedModel):
                getattr(self, name).reset()
            else:
                setattr(self, name, value)

    def asdict(self):
        """Convert a model to a dictionary."""
        warnings.warn(
            trans._(
                "The `asdict` method has been renamed `dict` and is now deprecated. It will be removed in 0.4.7",
                deferred=True,
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self.dict()

    def update(self, values):
        """Update a model in place.

        Parameters
        ----------
        values : dict, napari.utils.events.EventedModel
            Values to update the model with. If an EventedModel is passed it is
            first converted to a dictionary. The keys of this dictionary must
            be found as attributes on the current model.
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
                if isinstance(field, EventedModel):
                    field.update(value)
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
        if isinstance(other, EventedModel):
            for f_name, eq in self.__eq_operators__.items():
                if f_name not in other.__eq_operators__:
                    return False
                if not eq(getattr(self, f_name), getattr(other, f_name)):
                    return False
            return True
        else:
            return self.dict() == other

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
