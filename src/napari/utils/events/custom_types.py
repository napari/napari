from typing import Any

import numpy as np
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

# In numpy 2, the semantics of the copy argument in np.array changed
# so that copy=False errors if a copy is needed:
# https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
#
# In numpy 1, copy=False meant that a copy was avoided unless necessary,
# but would not error.
#
# In most usage like this use np.asarray instead, but sometimes we need
# to use some of the unique arguments of np.array (e.g. ndmin).
#
# This solution assumes numpy 1 by default, and switches to the numpy 2
# value for any release of numpy 2 on PyPI (including betas and RCs).
copy_if_needed: bool | None = False
if np.lib.NumpyVersion(np.__version__) >= '2.0.0b1':
    copy_if_needed = None


class Array(np.ndarray):
    """A numpy array type that works with Pydantic V2 validation."""

    def __class_getitem__(cls, t):
        return type('Array', (Array,), {'__dtype__': t})

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Define how Pydantic V2 should validate this type."""
        return core_schema.no_info_before_validator_function(
            cls.validate_type,
            core_schema.any_schema(),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Define the JSON schema for this type."""
        return {'type': 'array'}

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


class NumberNotEqError(ValueError):
    """Error raised when a number equals a prohibited value."""

    def __init__(self, prohibited: 'int | float') -> None:
        self.prohibited = prohibited
        super().__init__(f'ensure this value is not equal to {prohibited}')


class ConstrainedInt(int):
    """ConstrainedInt extension that adds not-equal.

    In Pydantic V2, use Annotated[int, Field(gt=..., lt=...)] for most constraints.
    This class is kept for backward compatibility with the 'ne' constraint.
    """

    strict: bool = False
    gt: int | None = None
    ge: int | None = None
    lt: int | None = None
    le: int | None = None
    multiple_of: int | None = None
    ne: int | list[int] | None = None

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Define how Pydantic V2 should validate this type."""
        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.int_schema(
                strict=cls.strict,
                gt=cls.gt,
                ge=cls.ge,
                lt=cls.lt,
                le=cls.le,
                multiple_of=cls.multiple_of,
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Define the JSON schema for this type."""
        schema: dict[str, Any] = {'type': 'integer'}
        if cls.gt is not None:
            schema['exclusiveMinimum'] = cls.gt
        if cls.ge is not None:
            schema['minimum'] = cls.ge
        if cls.lt is not None:
            schema['exclusiveMaximum'] = cls.lt
        if cls.le is not None:
            schema['maximum'] = cls.le
        if cls.multiple_of is not None:
            schema['multipleOf'] = cls.multiple_of
        if cls.ne is not None:
            f = 'const' if isinstance(cls.ne, int) else 'enum'
            schema['not'] = {f: cls.ne}
        return schema

    @classmethod
    def _validate(cls, v: int) -> int:
        _ne = cls.ne
        if _ne is not None and v in (_ne if isinstance(_ne, list) else [_ne]):
            raise NumberNotEqError(prohibited=cls.ne)
        return v


def conint(
    *,
    strict: bool = False,
    gt: int | None = None,
    ge: int | None = None,
    lt: int | None = None,
    le: int | None = None,
    multiple_of: int | None = None,
    ne: int | None = None,
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


class ConstrainedFloat(float):
    """ConstrainedFloat extension that adds step size.

    In Pydantic V2, use Annotated[float, Field(gt=..., lt=...)] for most constraints.
    This class is kept for backward compatibility with the 'step' constraint.
    """

    strict: bool = False
    gt: float | None = None
    ge: float | None = None
    lt: float | None = None
    le: float | None = None
    multiple_of: float | None = None
    step: float | None = None

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Define how Pydantic V2 should validate this type."""
        return core_schema.no_info_before_validator_function(
            cls._validate,
            core_schema.float_schema(
                strict=cls.strict,
                gt=cls.gt,
                ge=cls.ge,
                lt=cls.lt,
                le=cls.le,
                multiple_of=cls.multiple_of,
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Define the JSON schema for this type."""
        schema: dict[str, Any] = {'type': 'number'}
        if cls.gt is not None:
            schema['exclusiveMinimum'] = cls.gt
        if cls.ge is not None:
            schema['minimum'] = cls.ge
        if cls.lt is not None:
            schema['exclusiveMaximum'] = cls.lt
        if cls.le is not None:
            schema['maximum'] = cls.le
        if cls.multiple_of is not None:
            schema['multipleOf'] = cls.multiple_of
        if cls.step is not None:
            schema['step'] = cls.step
        return schema

    @classmethod
    def _validate(cls, v: Any) -> float:
        if not isinstance(v, (int, float)):
            raise TypeError('float required')
        return float(v)


def confloat(
    *,
    strict: bool = False,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
    multiple_of: float | None = None,
    step: float | None = None,
) -> type[float]:
    """Extended version of `pydantic.types.confloat` that includes step size."""
    # use kwargs then define conf in a dict to aid with IDE type hinting
    namespace = {
        'strict': strict,
        'gt': gt,
        'ge': ge,
        'lt': lt,
        'le': le,
        'multiple_of': multiple_of,
        'step': step,
    }
    return type('ConstrainedFloatValue', (ConstrainedFloat,), namespace)
