import operator
import sys
import warnings
from contextlib import contextmanager
from typing import Any, Callable, ClassVar, Dict, Optional, Set, Tuple, Union

import numpy as np
from pydantic import (
    BaseModel,
    PrivateAttr,
    main,
    utils,
    validate_model,
    validator,
)

from ...utils.events.containers import (
    EventedDict,
    EventedList,
    EventedSet,
    View,
)
from ...utils.misc import pick_equality_operator
from ..translations import trans
from .event import EmitterGroup, Event
from .evented import EventedMutable

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
        # check for properties defined on the class, so we can allow them
        # in EventedModel.__setattr__ and eventually enable events for them
        cls.__properties__ = {}
        for name, attr in namespace.items():
            if isinstance(attr, property):
                cls.__properties__[name] = attr
        comp_fields, dependencies = _get_computed_fields(cls)
        cls.__computed_fields__ = comp_fields
        cls.__field_dependencies__ = dependencies
        return cls


def _get_computed_fields(cls: 'EventedModel') -> Dict[str, Set[str]]:
    """Return list of computed fields and mapping of dependencies.

    Computed fields may be declared in the Model Config in order to add and
    hook up events for them, as well as wrapping the return value in Views
    if the computed field has a property setter.

    For example, if @property 'c' depends on model fields 'a' and 'b':

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
                computed_fields = {'c': ['a', 'b']}
    """
    dependency_map: Dict[str, Set[str]] = {}
    computed_fields = getattr(cls.__config__, 'computed_fields', {})
    for prop, fields in computed_fields.items():
        if prop not in cls.__properties__:
            raise ValueError(
                'Fields with dependencies must be properties. '
                f'{prop!r} is not.'
            )
        for field in fields:
            if field not in cls.__fields__:
                warnings.warn(f"Unrecognized field dependency: {field}")
            dependency_map.setdefault(field, set()).add(prop)
    return list(computed_fields), dependency_map


class EventedModel(BaseModel, metaclass=EventedMetaclass):
    """A Model subclass that emits an event whenever a field value is changed.

    Note: As per the standard pydantic behavior, default Field values are
    not validated (#4138) and should be correctly typed.
    """

    # add private attributes for event emission
    _events: EmitterGroup = PrivateAttr(default_factory=EmitterGroup)
    _parent: Optional[Tuple['EventedModel', str]] = PrivateAttr(None)
    _validate: bool = PrivateAttr(True)

    # mapping of name -> property obj for methods that are property setters
    __properties__: ClassVar[Dict[str, property]]
    # mapping of field name -> dependent set of property names
    # when field is changed, an event for dependent properties will be emitted.
    __field_dependencies__: ClassVar[Dict[str, Set[str]]]
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
        # note that this should stay True for Field(allow_mutation) to work.
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
        # this custom use of allow mutation behaves as normal (0 is false, 1 and 2 are true)
        # but allows us to use 2 as a special value meaning "inplace mutation"
        allow_mutation = 2

    @validator('*', pre=True, always=True)
    def _no_evented_collections(v):
        # we need to sanitize inputs to avoid loops of validation (if input is
        # EventedList, changing its content during validation will cause a mess)
        # this is very important because we may trigger validations with partial states
        # which will cause the model to revert to "usable" conditions
        if isinstance(v, (EventedList, EventedDict, EventedSet)):
            return v._uneventful()
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parent = None
        self._validate = True

        self._events.source = self

        # add event emitters for each field which is mutable
        field_events = [
            name
            for name, field in self.__fields__.items()
            if field.field_info.allow_mutation
        ]

        self._events.add(
            **dict.fromkeys(field_events + list(self.__computed_fields__))
        )

        # hook up events and parent validator for children
        for name in field_events:
            child = getattr(self, name)
            if isinstance(child, EventedMutable):
                # while seemingly redundant, this next line is very important to maintain
                # correct sources; see https://github.com/napari/napari/pull/4138
                # we solve it by re-setting the source after initial validation, which allows
                # us to use `validate_all = True`
                child.events.source = child
                child._parent = (self, name)
                # TODO: won't track sources all the way in?
                child.events.connect(getattr(self.events, name))

    def _pre_validate(self, values):
        if not self._validate:
            return values
        values, _, error = validate_model(self.__class__, values)
        if error:
            raise error
        if self._parent is not None:
            parent = self._parent[0]
            field = self._parent[1]
            pdict = parent.dict()
            pdict[field] = values
            values = parent._pre_validate(pdict)[field].dict()
        return values

    @contextmanager
    def _no_validation(self):
        self._validate = False
        yield
        self._validate = True

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        # avoid recursion
        if name in ('__computed_fields__', '__properties__'):
            return attr
        if (
            name in self.__computed_fields__
            and self.__properties__[name].fset is not None
        ):
            return View(attr, attr=name, parent=self)
        return attr

    def _super_setattr_(self, name: str, value: Any) -> None:
        # pydantic will raise a ValueError if extra fields are not allowed
        # so we first check to see if this field has a property.setter.
        # if so, we use it instead.
        if name in self.__properties__:
            if self.__properties__[name].fset is None:
                raise AttributeError(f"can't set attribute '{name}'")
            if name in self.__computed_fields__:
                # block events during this or we could have many duplicates, especially
                # if the setter sets several interconnected fields
                # they will be fired later by __setattr__ as appropriate
                with getattr(self.events, name).blocker():
                    self.__properties__[name].fset(self, value)
            else:
                # good old normal property setter
                self.__properties__[name].fset(self, value)
        elif name in self.__fields__ and (
            self.__config__.allow_mutation == 2
            or self.__fields__[name].field_info.allow_mutation == 2
        ):
            # do inplace_mutation if possible
            field_value = getattr(self, name)
            if isinstance(field_value, EventedMutable):
                fields = self.dict()
                fields.update({name: value})
                valid = self._pre_validate(fields)
                field_value._update_inplace(valid[name])
            else:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in getattr(self, 'events', {}):
            # fallback to default behavior
            self._super_setattr_(name, value)
            return

        # grab current value for field and its dependent properties, if any
        before = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            for field in [name, *self.__field_dependencies__.get(name, {})]:
                before[field] = getattr(self, field, object())

        # set value using original setter
        self._super_setattr_(name, value)

        # if different we emit the event with new value
        after = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            for field in [name, *self.__field_dependencies__.get(name, {})]:
                after[field] = getattr(self, field, object())

        for field in before:
            are_equal = self.__eq_operators__.get(field, operator.eq)
            if not are_equal(after[field], before[field]):
                getattr(self.events, field)(value=after[field])  # emit event

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

        NOTE: this does NOT trigger individual events, but only a general model event

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

        # immutable fields would be added by validation, so we need to complain if the exist now,
        # and then remove them later!
        for k in values:
            if not self.__fields__[k].field_info.allow_mutation and values[
                k
            ] != getattr(self, k):
                raise TypeError(
                    f'"{k}" has allow_mutation set to False and cannot be assigned'
                )
        values = self._pre_validate(values)
        values = {
            k: v
            for k, v in values.items()
            if self.__fields__[k].field_info.allow_mutation
        }

        with self.events.blocker() as block:
            with self._no_validation():
                for key, value in values.items():
                    field = getattr(self, key)
                    if field == value:
                        continue
                    if isinstance(field, EventedMutable) and recurse:
                        field._update_inplace(value)
                    else:
                        # validation is all good, so let's jump pydantic!
                        self.__dict__[key] = value
                        getattr(self.events, key)(value=value)

        # TODO: shouldn't we still trigger events for all the fields?
        if block.count:
            self.events(Event(self))

    def _update_inplace(self, other):
        with self._no_validation():
            self.update(other)

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
