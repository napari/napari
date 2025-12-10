import numpy as np
from app_model.types import KeyBinding
from psygnal import EventedModel as EventedModel_
from pydantic import ConfigDict

# encoders for non-napari specific field types.  To declare a custom encoder
# for a napari type, add a `_json_encode` method to the class itself.
# it will be added to the model json_encoders in :func:`EventedMetaclass.__new__`
_BASE_JSON_ENCODERS = {
    np.ndarray: lambda arr: arr.tolist(),
    KeyBinding: lambda v: str(v),
}


class EventedModel(EventedModel_):
    # pydantic BaseModel configuration.  see:
    # https://pydantic-docs.helpmanual.io/usage/model_config/
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        validate_default=True,
        json_encoders=_BASE_JSON_ENCODERS,
    )
