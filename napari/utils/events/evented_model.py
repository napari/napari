import operator
import warnings
from typing import Any, Callable, ClassVar, Dict, Set

from pydantic import BaseModel, PrivateAttr
from pydantic.main import ModelMetaclass

from ...utils.misc import pick_equality_operator
from .custom_types import JSON_ENCODERS
from .event import EmitterGroup, Event


class EqualityMetaclass(ModelMetaclass):
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
        cls.__eq_operators__ = {
            n: pick_equality_operator(f.type_)
            for n, f in cls.__fields__.items()
        }
        return cls


class EventedModel(BaseModel, metaclass=EqualityMetaclass):

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
        # whether to populate models with the value property of enums, rather
        # than the raw enum. This may be useful if you want to serialise
        # model.dict() later
        use_enum_values = True
        # whether to validate field defaults (default: False)
        validate_all = True
        # a dict used to customise the way types are encoded to JSON
        # https://pydantic-docs.helpmanual.io/usage/exporting_models/#modeljson
        json_encoders = JSON_ENCODERS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # add events for each field
        self._events.source = self
        self._events.add(**dict.fromkeys(self.__fields__))

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

    def asdict(self):
        """Convert a model to a dictionary."""
        warnings.warn(
            (
                "The `asdict` method has been renamed `dict` and is now "
                "deprecated. It will be removed in 0.4.7"
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
            raise ValueError(f"Unsupported update from {type(values)}")

        with self.events.blocker() as block:
            for key, value in values.items():
                setattr(self, key, value)

        if block.count:
            self.events(Event(self))

    def __eq__(self, other) -> bool:
        """Check equality with another object.

        We override the pydantic approach (which just checks
        ``self.dict() == other.dict()``) to accomodate more complicated types
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
