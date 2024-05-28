from collections.abc import Generator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Union,
)

import numpy as np

from napari._pydantic_compat import errors, types

if TYPE_CHECKING:
    from decimal import Decimal

    from napari._pydantic_compat import ModelField

    Number = Union[int, float, Decimal]

# In numpy 2, the semantics of the copy argument in np.array changed:
# https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
#
# We would normally use np.asarray instead, but sometimes we need
# to use some of the unique arguments of np.array (e.g. ndmin).
#
# This solution is vendored from scipy:
# https://github.com/scipy/scipy/blob/c1c3738f57940f2dd7e3453c428b20e031d7a02e/scipy/_lib/_util.py#L59
copy_if_needed: Optional[bool]

if np.lib.NumpyVersion(np.__version__) >= '2.0.0':
    copy_if_needed = None
elif np.lib.NumpyVersion(np.__version__) < '1.28.0':
    copy_if_needed = False
else:
    # 2.0.0 dev versions, handle cases where copy may or may not exist
    try:
        np.array([1]).__array__(copy=None)  # type: ignore[call-overload]
        copy_if_needed = None
    except TypeError:
        copy_if_needed = False


class Array(np.ndarray):
    def __class_getitem__(cls, t):
        return type('Array', (Array,), {'__dtype__': t})

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        dtype = getattr(cls, '__dtype__', None)
        if isinstance(dtype, tuple):
            dtype, shape = dtype
        else:
            shape = ()

        result = np.array(
            val, dtype=dtype, copy=copy_if_needed, ndmin=len(shape)
        )

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

    ne: Optional[Union[int, list[int]]] = None

    @classmethod
    def __modify_schema__(cls, field_schema: dict[str, Any]) -> None:
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
    gt: Optional[int] = None,
    ge: Optional[int] = None,
    lt: Optional[int] = None,
    le: Optional[int] = None,
    multiple_of: Optional[int] = None,
    ne: Optional[int] = None,
) -> type[int]:
    """Extended version of `pydantic.types.conint` that includes not-equal."""
    # use kwargs then define conf in a dict to aid with IDE type hinting
    namespace = {
        'strict': strict,
        'gt': gt,
        'ge': ge,
        'lt': lt,
        'le': le,
        'multiple_of': multiple_of,
        'ne': ne,
    }
    return type('ConstrainedIntValue', (ConstrainedInt,), namespace)
