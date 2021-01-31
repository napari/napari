import warnings
from typing import ClassVar, Dict, Set

from pydantic import BaseModel, PrivateAttr

from .custom_types import JSON_ENCODERS
from .dataclass import _type_to_compare, is_equal
from .event import EmitterGroup


class EventedModel(BaseModel):

    # add private attributes for event emission
    _events: EmitterGroup = PrivateAttr(default_factory=EmitterGroup)
    __equality_checks__: Dict = PrivateAttr(default_factory=dict)
    __slots__: ClassVar[Set[str]] = {"__weakref__"}

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

        # create dict with compare functions for fields which cannot be compared
        # using standard equal operator, like numpy arrays.
        compare_dict = {
            field.name: _type_to_compare(field.type_)
            for field in self.__fields__.values()
            if _type_to_compare(field.type_) is not None
        }

        self.__equality_checks__.update(compare_dict)

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
        if not self.__equality_checks__.get(name, is_equal)(after, before):
            # emit event
            getattr(self.events, name)(value=after)

    # expose the private EmitterGroup publically
    @property
    def events(self):
        return self._events

    def asdict(self):
        """Convert a model to a dictionary."""
        warnings.warn(
            (
                "The `asdict` method has been renamed `dict` and is now deprecated. It will be"
                " removed in 0.4.7"
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
            Values to update the model with. If an EventedModel is passed it is first
            converted to a dictionary. The keys of this dictionary must be found as
            attributes on the current model.
        """
        if isinstance(values, self.__class__):
            values = values.dict()
        if not isinstance(values, dict):
            raise ValueError(f"Unsupported update from {type(values)}")

        for key, value in values.items():
            self.key = value
