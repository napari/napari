from decimal import Decimal
from typing import Any, Callable, Dict, Generator, List, Optional, Type, Union

import dask.array as da
import numpy as np
import numpy.typing as npt
import zarr
from pydantic import errors, types
from pydantic.fields import ModelField as ModelField

Number = Union[int, float, Decimal]

Array = Union[npt.ArrayLike, np.ndarray, da.Array, zarr.Array]

class NumberNotEqError(errors.PydanticValueError):
    code: str = ...
    msg_template: str = ...
    def __init__(self, prohibited: Number) -> None: ...

class ConstrainedInt(types.ConstrainedInt):
    ne: Optional[Union[int, List[int]]] = ...
    @classmethod
    def __modify_schema__(cls: Any, field_schema: Dict[str, Any]) -> None: ...
    @classmethod
    def __get_validators__(
        cls: Any,
    ) -> Generator[Callable[..., Any], None, None]: ...
    @staticmethod
    def validate_ne(v: Number, field: ModelField) -> Number: ...

def conint(
    *,
    strict: bool = ...,
    gt: int = ...,
    ge: int = ...,
    lt: int = ...,
    le: int = ...,
    multiple_of: int = ...,
    ne: int = ...,
) -> Type[int]: ...
