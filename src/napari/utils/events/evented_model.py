import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, ClassVar, Union

from typing_extensions import Self

import numpy as np
from app_model.types import KeyBinding

from napari._pydantic_compat import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    field_serializer,
)
from napari.utils.events.event import EmitterGroup, Event
from napari.utils.misc import pick_equality_operator
from napari.utils.translations import trans


def _get_field_names(cls: type['EventedModel']) -> set[str]:
    """Get the set of field names from annotations (excluding private attributes)."""
    field_names = set()
    for klass in cls.__mro__:
        if klass is EventedModel:
            break
        for name in getattr(klass, '__annotations__', {}):
            if not name.startswith('_'):
                field_names.add(name)
    return field_names


def _get_field_dependents(cls: type['EventedModel']) -> dict[str, set[str]]:
    """Return mapping of field name -> dependent set of property names.

    Dependencies will be guessed by inspecting the code of each property
    in order to emit an event for a computed property when a model field
    that it depends on changes (e.g: @property 'c' depends on model fields
    'a' and 'b'). Alternatively, dependencies may be declared explicitly
    in the model_config.

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
    # Get field names from annotations since model_fields is not populated yet in __init_subclass__
    field_names = _get_field_names(cls)

    # Get dependencies from model_config if available
    _deps = cls.model_config.get('dependencies') if hasattr(cls, 'model_config') else None
    if _deps:
        for prop_name, fields in _deps.items():
            if prop_name not in cls.__properties__:
                raise ValueError(
                    'Fields with dependencies must be properties. '
                    f'{prop_name!r} is not.'
                )
            for field in fields:
                if field not in field_names:
                    warnings.warn(f'Unrecognized field dependency: {field}')
                deps.setdefault(field, set()).add(prop_name)
    else:
        # if dependencies haven't been explicitly defined, we can glean
        # them from the property.fget code object:
        for prop_name, prop in cls.__properties__.items():
            _update_dependents_from_property_code(cls, prop_name, prop, deps, field_names)
    return deps


def _update_dependents_from_property_code(
    cls, prop_name, prop, deps, field_names, visited=()
):
    """Recursively find all the dependents of a property by inspecting the code object.

    Update the given deps dictionary with the new findings.
    """
    for name in prop.fget.__code__.co_names:
        if name in field_names:
            deps.setdefault(name, set()).add(prop_name)
        elif name in cls.__properties__ and name not in visited:
            # to avoid infinite recursion, we shouldn't re-check getter we've already seen
            visited = visited + (name,)
            # sub_prop is the new property, but we leave prop_name the same
            sub_prop = cls.__properties__[name]
            _update_dependents_from_property_code(
                cls, prop_name, sub_prop, deps, field_names, visited
            )


class EventedModel(BaseModel):
    """A Model subclass that emits an event whenever a field value is changed.

    Note: As per the standard pydantic behavior, default Field values are
    not validated (#4138) and should be correctly typed.
    """

    # Pydantic V2 configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        # In V2, extra fields handling is via 'extra' key
        extra='ignore',
        # revalidate_instances is the V2 way to ensure re-validation
        revalidate_instances='always',
    )

    # add private attributes for event emission
    _events: EmitterGroup = PrivateAttr(default_factory=EmitterGroup)

    # mapping of name -> property obj for methods that are properties
    __properties__: ClassVar[dict[str, property]] = {}
    # mapping of field name -> dependent set of property names
    # when field is changed, an event for dependent properties will be emitted.
    __field_dependents__: ClassVar[dict[str, set[str]]] = {}
    __eq_operators__: ClassVar[dict[str, Callable[[Any, Any], bool]]] = {}
    _changes_queue: dict[str, Any] = PrivateAttr(default_factory=dict)
    _primary_changes: set[str] = PrivateAttr(default_factory=set)
    _delay_check_semaphore: int = PrivateAttr(0)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Set up equality operators and properties when subclass is created.

        This replaces the V1 EventedMetaclass functionality.
        """
        super().__init_subclass__(**kwargs)

        # Initialize class variables
        cls.__eq_operators__ = {}
        cls.__properties__ = {}

        # Set up equality operators for each field using __annotations__
        # In Pydantic V2, model_fields isn't populated until after __init_subclass__,
        # so we use __annotations__ directly.
        for field_name, annotation in getattr(cls, '__annotations__', {}).items():
            if not field_name.startswith('_'):  # Skip private attributes
                cls.__eq_operators__[field_name] = pick_equality_operator(annotation)

        # Check for properties defined on the class (not inherited from BaseModel/EventedModel)
        # Walk the MRO but stop at EventedModel to exclude BaseModel properties
        for klass in cls.__mro__:
            if klass is EventedModel:
                break
            for name, attr in vars(klass).items():
                if isinstance(attr, property) and name not in cls.__properties__:
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

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._events.source = self
        # add event emitters for each field which is mutable (not frozen)
        field_events = []
        for name, field_info in type(self).model_fields.items():
            # In V2, check if field is frozen
            is_frozen = field_info.frozen if field_info.frozen is not None else False
            if not is_frozen:
                field_events.append(name)

        self._events.add(
            **dict.fromkeys(field_events + list(self.__properties__))
        )

        # while seemingly redundant, this next line is very important to maintain
        # correct sources; see https://github.com/napari/napari/pull/4138
        # we solve it by re-setting the source after initial validation, which allows
        # us to use revalidate_instances='always'
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
            self._primary_changes.add(name)
            self._setattr_impl(name, value)

    def _check_if_values_changed_and_emit_if_needed(self):
        """
        Check if field values changed and emit events if needed.

        The advantage of moving this to the end of all the modifications is
        that comparisons will be performed only once for every potential change.
        """
        if self._delay_check_semaphore > 0 or len(self._changes_queue) == 0:
            # do not run whole machinery if there is no need
            return
        to_emit = []
        for name in self._primary_changes:
            # primary changes should contains only fields that are changed directly by assignment
            if name not in self._changes_queue:
                continue
            old_value = self._changes_queue[name]
            if (res := self._check_if_differ(name, old_value))[0]:
                to_emit.append((name, res[1]))
            self._changes_queue.pop(name)
        if not to_emit:
            # If no direct changes was made then we can skip whole machinery
            self._changes_queue.clear()
            self._primary_changes.clear()
            return
        for name, old_value in self._changes_queue.items():
            # check if any of dependent properties changed
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
        for name in type(self).model_fields:
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
            if isinstance(value, EventedModel):
                getattr(self, name).reset()
            else:
                # In V2, check model_config for frozen and field for frozen
                model_frozen = self.model_config.get('frozen', False)
                field_info = type(self).model_fields.get(name)
                field_frozen = field_info.frozen if field_info and field_info.frozen is not None else False
                if not model_frozen and not field_frozen:
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
        is constructed in ``__init_subclass__``
        """
        if self is other:
            return True
        if not isinstance(other, EventedModel):
            return self.model_dump() == other
        if self.__class__ != other.__class__:
            return False
        for f_name in type(self).model_fields:
            eq = self.__eq_operators__.get(f_name, lambda a, b: a == b)
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
        # In V2, this is handled via model_dump(mode='python') vs model_dump()
        # We keep this for API compatibility but it's a no-op in V2
        yield

    # V2 serializers for common types
    @field_serializer('*', mode='wrap')
    @classmethod
    def _serialize_any(cls, value: Any, handler: Callable) -> Any:
        """Handle serialization of numpy arrays and other special types."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, KeyBinding):
            return str(value)
        # Check if the type has a custom _json_encode method
        if hasattr(type(value), '_json_encode'):
            return type(value)._json_encode(value)
        return handler(value)


def get_defaults(obj: BaseModel) -> dict[str, Any]:
    """Get possibly nested default values for a Model object."""
    from pydantic.fields import PydanticUndefined

    dflt = {}
    # Use type(obj).model_fields to avoid deprecation warning in Pydantic V2.11+
    for k, field_info in type(obj).model_fields.items():
        d = field_info.default
        # Handle default_factory (used when default is PydanticUndefined)
        if d is PydanticUndefined and field_info.default_factory is not None:
            d = field_info.default_factory()
        elif d is None and field_info.annotation is not None:
            # Check if the annotation is a BaseModel subclass
            try:
                if isinstance(field_info.annotation, type) and issubclass(field_info.annotation, BaseModel):
                    d = get_defaults(field_info.annotation)
            except TypeError:
                pass
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
