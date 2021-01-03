from typing import Dict

from pydantic import BaseModel, PrivateAttr

from ..events.dataclass import _type_to_compare, is_equal
from ..events.event import EmitterGroup
from .custom_types import JSON_ENCODERS


class EventedModel(BaseModel):

    # add private attributes for event emission
    _events: EmitterGroup = PrivateAttr(default_factory=EmitterGroup)
    __equality_checks__: Dict = PrivateAttr(default_factory=dict)

    # Definte the config so that assigments are validated
    # and add custom encoders
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

        # get fields and add to EmitterGroup
        _fields = list(self.__fields__)
        e_fields = {fld: None for fld in _fields}

        # add events for each field
        self.events.add(**e_fields)

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
