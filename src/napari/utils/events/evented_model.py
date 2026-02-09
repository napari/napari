import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, ClassVar, Union

import numpy as np
from app_model.types import KeyBinding
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
)
from pydantic._internal._model_construction import ModelMetaclass

from napari._pydantic_util import get_inner_type, get_outer_type
from napari.utils.events.event import EmitterGroup, Event
from napari.utils.misc import pick_equality_operator
from napari.utils.translations import trans

# encoders for non-napari specific field types.  To declare a custom encoder
# for a napari type, add a `_json_encode` method to the class itself.
# it will be added to the model json_encoders in :func:`EventedMetaclass.__new__`
_BASE_JSON_ENCODERS = {
    np.ndarray: lambda arr: arr.tolist(),
    KeyBinding: lambda v: str(v),
}


class EventedMetaclass(ModelMetaclass):
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
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        cls.__eq_operators__ = {}
        for n, f in cls.model_fields.items():
            field_type = get_outer_type(f.annotation)
            cls.__eq_operators__[n] = pick_equality_operator(field_type)
            # If a field type has a _json_encode method, add it to the json
            # encoders for this model.
            # NOTE: a _json_encode field must return an object that can be
            # passed to json.dumps ... but it needn't return a string.
            if hasattr(field_type, '_json_encode'):
                encoder = field_type._json_encode
                cls.model_config['json_encoders'][field_type] = encoder
                # also add it to the base config
                # required for pydantic>=1.8.0 due to:
                # https://github.com/samuelcolvin/pydantic/pull/2064
                EventedModel.model_config['json_encoders'][field_type] = (
                    encoder
                )
        # check for properties defined on the class, so we can allow them
        # in EventedModel.__setattr__ and create events
        cls.__properties__ = {}
        for name, attr in namespace.items():
            if isinstance(attr, property):
                cls.__properties__[name] = attr
                # determine compare operator
                if (
                    hasattr(attr.fget, '__annotations__')
                    and 'return' in attr.fget.__annotations__
                    and not isinstance(
                        attr.fget.__annotations__['return'], str
                    )
                ):
                    cls.__eq_operators__[name] = pick_equality_operator(
                        attr.fget.__annotations__['return']
                    )

        cls.__field_dependents__ = _get_field_dependents(cls)
        return cls


def _update_dependents_from_property_code(
    cls, prop_name, prop, deps, visited=()
):
    """Recursively find all the dependents of a property by inspecting the code object.

    Update the given deps dictionary with the new findings.
    """
    for name in prop.fget.__code__.co_names:
        if name in cls.model_fields:
            deps.setdefault(name, set()).add(prop_name)
        elif name in cls.__properties__ and name not in visited:
            # to avoid infinite recursion, we shouldn't re-check getter we've already seen
            visited = visited + (name,)
            # sub_prop is the new property, but we leave prop_name the same
            sub_prop = cls.__properties__[name]
            _update_dependents_from_property_code(
                cls, prop_name, sub_prop, deps, visited
            )


def _get_field_dependents(cls: 'EventedModel') -> dict[str, set[str]]:
    """Return mapping of field name -> dependent set of property names.

    Dependencies will be guessed by inspecting the code of each property
    in order to emit an event for a computed property when a model field
    that it depends on changes (e.g: @property 'c' depends on model fields
    'a' and 'b'). Alternatvely, dependencies may be declared excplicitly
    in the Model Config.

    Note: accessing a field with `getattr()` instead of dot notation won't
    be automatically detected.

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

            @property
            def d(self) -> int:
                return sum(self.c)

            @d.setter
            def d(self, val: int):
                self.c = [val // 2, val // 2]

            model_config = ConfigDict(
                dependencies={
                    'c': ['a', 'b'],
                    'd': ['a', 'b']
                }
                )
    """
    if not cls.__properties__:
        return {}

    deps: dict[str, set[str]] = {}

    _deps = cls.model_config.get('dependencies')
    if _deps:
        for prop_name, fields in _deps.items():
            if prop_name not in cls.__properties__:
                raise ValueError(
                    'Fields with dependencies must be properties. '
                    f'{prop_name!r} is not.'
                )
            for field in fields:
                if field not in cls.model_fields:
                    warnings.warn(f'Unrecognized field dependency: {field}')
                deps.setdefault(field, set()).add(prop_name)
    else:
        # if dependencies haven't been explicitly defined, we can glean
        # them from the property.fget code object:
        for prop_name, prop in cls.__properties__.items():
            _update_dependents_from_property_code(cls, prop_name, prop, deps)
    return deps


class EventedModel(BaseModel, metaclass=EventedMetaclass):
    """A Model subclass that emits an event whenever a field value is changed.

    Note: As per the standard pydantic behavior, default Field values are
    not validated (#4138) and should be correctly typed.
    """

    # add private attributes for event emission
    _events: EmitterGroup = PrivateAttr(default_factory=EmitterGroup)

    # mapping of name -> property obj for methods that are properties
    __properties__: ClassVar[dict[str, property]]
    # mapping of field name -> dependent set of property names
    # when field is changed, an event for dependent properties will be emitted.
    __field_dependents__: ClassVar[dict[str, set[str]]]
    __eq_operators__: ClassVar[dict[str, Callable[[Any, Any], bool]]]
    _changes_queue: dict[str, Any] = PrivateAttr(default_factory=dict)
    _primary_changes: dict[str, None] = PrivateAttr(default_factory=dict)
    _delay_check_semaphore: int = PrivateAttr(0)
    __slots__: ClassVar[set[str]] = {'__weakref__'}  # type: ignore

    # pydantic BaseModel configuration.  see:
    # https://pydantic-docs.helpmanual.io/usage/model_config/
    model_config = ConfigDict(
        # whether to allow arbitrary user types for fields (they are validated
        # simply by checking if the value is an instance of the type). If
        # False, RuntimeError will be raised on model declaration
        arbitrary_types_allowed=True,
        validate_assignment=True,
        validate_default=True,
        # https://pydantic-docs.helpmanual.io/usage/exporting_models/#modeljson
        # NOTE: json_encoders are also added EventedMetaclass.__new__ if the
        # field declares a _json_encode method.
        json_encoders=_BASE_JSON_ENCODERS,
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._events.source = self
        # add event emitters for each field which is mutable
        field_events = [
            name
            for name, field in self.__class__.model_fields.items()
            if not field.frozen
        ]

        self._events.add(
            **dict.fromkeys(field_events + list(self.__properties__))
        )

        # while seemingly redundant, this next line is very important to maintain
        # correct sources; see https://github.com/napari/napari/pull/4138
        # we solve it by re-setting the source after initial validation, which allows
        # us to use `validate_all = True`
        self._reset_event_source()

    def _super_setattr_(self, name: str, value: Any) -> None:
        # pydantic will raise a ValueError if extra fields are not allowed
        # so we first check to see if this field is a property
        # if so, we use it instead.
        if name in self.__properties__:
            setter = self.__properties__[name].fset
            if setter is None:
                # raise same error as normal properties
                raise AttributeError(f"can't set attribute '{name}'")
            setter(self, value)
        else:
            super().__setattr__(name, value)

    def _check_if_differ(self, name: str, old_value: Any) -> tuple[bool, Any]:
        """
        Check new value of a field and emit event if it is different from the old one.

        Returns True if data changed, else False. Return current value.
        """
        new_value = getattr(self, name, object())
        if name in self.__eq_operators__:
            are_equal = self.__eq_operators__[name]
        else:
            are_equal = pick_equality_operator(new_value)
        return not are_equal(new_value, old_value), new_value

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in getattr(self, 'events', {}):
            # This is a workaround needed because `EventedConfigFileSettings` uses
            # `_config_path` before calling the superclass constructor
            super().__setattr__(name, value)
            return
        with ComparisonDelayer(self):
            # NOTE: this is a dict and not just a set, because we need ORDERED!
            self._primary_changes[name] = None
            self._setattr_impl(name, value)

    def _check_if_values_changed_and_emit_if_needed(self):
        """
        Check if field values changed and emit events if needed.

        The advantage of moving this to the end of all the modifications is
        that comparisons will be performed only once for every potential change.
        """
        if self._delay_check_semaphore > 0 or len(self._changes_queue) == 0:
            # do not run whole machinery if there is no need
            if self._delay_check_semaphore == 0:
                # it means that we finished processing changes,
                # but there is no callback connected to events
                # associated with changed fields, so we need
                # to clear the queues to not have stale data on next changes
                self._primary_changes.clear()
            return
        to_emit = []
        for name in self._primary_changes:
            # primary changes should contain only fields that are changed directly by assignment
            if name not in self._changes_queue:
                continue
            old_value = self._changes_queue[name]
            if (res := self._check_if_differ(name, old_value))[0]:
                to_emit.append((name, res[1]))
            self._changes_queue.pop(name)
        if not to_emit:
            # If no direct changes were made then we can skip the whole machinery
            self._changes_queue.clear()
            self._primary_changes.clear()
            return
        for name, old_value in self._changes_queue.items():
            # check if any of the dependent properties changed
            if (res := self._check_if_differ(name, old_value))[0]:
                to_emit.append((name, res[1]))
        self._changes_queue.clear()
        self._primary_changes.clear()

        with ComparisonDelayer(self):
            # Again delay comparison to avoid having events caused by callback functions
            for name, new_value in to_emit:
                getattr(self.events, name)(value=new_value)

    def _setattr_impl(self, name: str, value: Any) -> None:
        if name not in getattr(self, 'events', {}):
            # fallback to default behavior
            self._super_setattr_(name, value)
            return

        # grab current value
        field_dep = self.__field_dependents__.get(name, set())
        has_callbacks = {
            name: bool(getattr(self.events, name).callbacks)
            for name in field_dep
        }
        emitter = getattr(self.events, name)
        # equality comparisons may be expensive, so just avoid them if
        # event has no callbacks connected
        if not (
            emitter.callbacks
            or self._events.callbacks
            or any(has_callbacks.values())
        ):
            self._super_setattr_(name, value)
            return

        dep_with_callbacks = [
            dep for dep, has_cb in has_callbacks.items() if has_cb
        ]

        if name not in self._changes_queue:
            self._changes_queue[name] = getattr(self, name, object())

        for dep in dep_with_callbacks:
            if dep not in self._changes_queue:
                self._changes_queue[dep] = getattr(self, dep, object())

        # set value using original setter
        self._super_setattr_(name, value)

    # expose the private EmitterGroup publicly
    @property
    def events(self) -> EmitterGroup:
        return self._events

    def _reset_event_source(self):
        """
        set the event sources of self and all the children to the correct values
        """
        # events are all messed up due to objects being probably
        # recreated arbitrarily during validation
        self.events.source = self
        for name in self.__class__.model_fields:
            child = getattr(self, name)
            if isinstance(child, EventedModel):
                # TODO: this isinstance check should be EventedMutables in the future
                child._reset_event_source()
            elif name in self.events.emitters:
                getattr(self.events, name).source = self

    @property
    def _defaults(self):
        return get_defaults(self)

    def reset(self):
        """Reset the state of the model to default values."""
        for name, value in self._defaults.items():
            if isinstance(getattr(self, name), EventedModel):
                getattr(self, name).reset()
            elif (
                not self.model_config.get('frozen', False)
                and not self.__class__.model_fields[name].frozen
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
            values = values.model_dump()
        if not isinstance(values, dict):
            raise TypeError(
                trans._(
                    'Unsupported update from {values}',
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
        ``self.model_dump() == other.model_dump()``) to accommodate more complicated types
        like arrays, whose truth value is often ambiguous. ``__eq_operators__``
        is constructed in ``EqualityMetaclass.__new__``
        """
        if self is other:
            return True
        if not isinstance(other, EventedModel):
            return self.model_dump() == other
        if self.__class__ != other.__class__:
            return False
        for f_name in self.__class__.model_fields:
            eq = self.__eq_operators__[f_name]
            if not eq(getattr(self, f_name), getattr(other, f_name)):
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
        before = self.model_config.get('use_enum_values', null)
        self.model_config['use_enum_values'] = as_values
        try:
            yield
        finally:
            if before is not null:
                self.model_config['use_enum_values'] = before
            else:
                del self.model_config['use_enum_values']

    @classmethod
    def model_json_schema(
        cls,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate a JSON schema for this model.

        This is required to prevent from mail formated docstring break docs build.
        """
        return super().model_json_schema(
            *args,
            **kwargs,
        )


def get_defaults(obj: BaseModel | type[BaseModel]) -> dict[str, Any]:
    """Get possibly nested default values for a Model object."""
    dflt = {}
    if isinstance(obj, BaseModel):
        model_fields = obj.__class__.model_fields
    else:
        model_fields = obj.model_fields
    for k, v in model_fields.items():
        if v.exclude:
            continue
        d = v.get_default()
        field_type = get_inner_type(v.annotation)
        if d is None:
            if isinstance(field_type, ModelMetaclass):
                d = get_defaults(field_type)
            else:
                try:
                    # pydantic require `validated_data` if `call_default_factory` is True
                    d = v.get_default(
                        call_default_factory=True, validated_data={}
                    )
                except TypeError:
                    # pydantic 2.9 and 2.8 do not allow passing `validated_data` to `get_default`,
                    # pydantic 2.10 requires it, so we need to catch the TypeError and try again
                    # without it for compatibility with 2.8 and 2.9
                    d = v.get_default(call_default_factory=True)

        dflt[k] = d
    return dflt


class ComparisonDelayer:
    def __init__(self, target: EventedModel):
        self._target = target

    def __enter__(self):
        self._target._delay_check_semaphore += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._target._delay_check_semaphore -= 1
        self._target._check_if_values_changed_and_emit_if_needed()
