from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Type,
    Union,
)

import numpy as np
from pydantic import errors, types

if TYPE_CHECKING:
    from decimal import Decimal

    from pydantic.fields import ModelField

    Number = Union[int, float, Decimal]


class Array(np.ndarray):
    _coerce_type = False

    def __class_getitem__(cls, t):
        return type('Array', (Array,), {'__dtype__': t})

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val, field=None):
        # we have to explicitly allow None when field.allow_none is True
        # because this is called before type coercion (which normally does this job).
        # Also, field=None is necessary because this validator is sometimes called manually
        if field is not None and field.allow_none and val is None:
            return None

        dtype = getattr(cls, '__dtype__', None)
        if isinstance(dtype, tuple):
            dtype, shape = dtype
        else:
            shape = tuple()

        result = np.array(val, dtype=dtype, copy=False, ndmin=len(shape))

        if any(
            (shape[i] != -1 and shape[i] != result.shape[i])
            for i in range(len(shape))
        ):
            result = result.reshape(shape)
        return result


class NumberNotEqError(errors.PydanticValueError):
    code = 'number.not_eq'
    msg_template = 'ensure this value is not equal to {prohibited}'

    def __init__(self, *, prohibited: 'Number') -> None:
        super().__init__(prohibited=prohibited)


class ConstrainedInt(types.ConstrainedInt):
    """ConstrainedInt extension that adds not-equal"""

    ne: Optional[Union[int, List[int]]] = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        super().__modify_schema__(field_schema)
        if cls.ne is not None:
            f = 'const' if isinstance(cls.ne, int) else 'enum'
            field_schema['not'] = {f: cls.ne}

    @classmethod
    def __get_validators__(cls) -> Generator[Callable[..., Any], None, None]:
        yield from super().__get_validators__()
        yield cls.validate_ne

    @staticmethod
    def validate_ne(v: 'Number', field: 'ModelField') -> 'Number':
        field_type: ConstrainedInt = field.type_
        _ne = field_type.ne
        if _ne is not None and v in (_ne if isinstance(_ne, list) else [_ne]):
            raise NumberNotEqError(prohibited=field_type.ne)
        return v


def conint(
    *,
    strict: bool = False,
    gt: int = None,
    ge: int = None,
    lt: int = None,
    le: int = None,
    multiple_of: int = None,
    ne: int = None,
) -> Type[int]:
    """Extended version of `pydantic.types.conint` that includes not-equal."""
    # use kwargs then define conf in a dict to aid with IDE type hinting
    namespace = dict(
        strict=strict,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        multiple_of=multiple_of,
        ne=ne,
        _coerce_type=False,
    )

    return type('ConstrainedIntValue', (ConstrainedInt,), namespace)
