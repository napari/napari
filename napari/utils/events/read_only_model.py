from pydantic import BaseModel


class ReadOnlyModel(BaseModel):
    """A pydantic model that is read only.

    This model is useful for making faux immutable objects like events
    which should not be modified by their callbacks.
    """

    class Config:
        # Once created the event is read only and not able to be modified
        allow_mutation = False
        # whether to populate models with the value property of enums, rather
        # than the raw enum. This may be useful if you want to serialise
        # model.dict() later
        use_enum_values = True
        # whether to validate field defaults (default: False)
        validate_all = True
