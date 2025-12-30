"""
Pydantic V2 compatibility module for napari.

This module provides the Pydantic V2 API for napari, with some compatibility
shims for code that was written using Pydantic V1 patterns.

Migration from V1 to V2:
- BaseModel: use directly from pydantic
- BaseSettings: now in pydantic-settings package
- validator -> field_validator (with @classmethod, mode='before'/'after')
- root_validator -> model_validator (mode='before'/'after')
- class Config -> model_config = ConfigDict(...)
- __fields__ -> model_fields
- .dict() -> .model_dump()
- .json() -> .model_dump_json()
- .parse_obj() -> .model_validate()
- parse_obj_as() -> TypeAdapter(T).validate_python()
"""

from typing import TYPE_CHECKING, Any, TypeVar

# Core Pydantic V2 imports
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.fields import FieldInfo
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

# BaseSettings is now in a separate package
from pydantic_settings import BaseSettings, SettingsConfigDict

# Color type moved to pydantic-extra-types
try:
    from pydantic_extra_types.color import Color
except ImportError:
    # Fallback if pydantic-extra-types not installed
    Color = None  # type: ignore[misc, assignment]

# Type variable for generic models
T = TypeVar('T')

# FilePath is now just a constrained Path
from pydantic import FilePath

# Annotated types for validation
from typing import Annotated

# For constrained types, use Annotated with Field
from pydantic import conlist, constr

# For sequence_like functionality
def sequence_like(v: Any) -> bool:
    """Check if a value is sequence-like (but not a string or bytes)."""
    return isinstance(v, (list, tuple, set, frozenset))

# ROOT_KEY equivalent (used in some validation contexts)
ROOT_KEY = '__root__'

# ---- Compatibility aliases for migration ----

# These are deprecated and will be removed. Use the V2 equivalents.
validator = field_validator  # deprecated: use @field_validator
root_validator = model_validator  # deprecated: use @model_validator


def parse_obj_as(type_: type[T], obj: Any) -> T:
    """Parse an object as a given type.

    Deprecated: Use TypeAdapter(type_).validate_python(obj) instead.
    """
    return TypeAdapter(type_).validate_python(obj)


# ---- Removed V1 APIs with stubs/alternatives ----

# ModelMetaclass is no longer used in V2. Instead, use:
# - __pydantic_complete__ class method for post-model-creation hooks
# - model_post_init for per-instance initialization
# We provide a stub that just returns the original metaclass
class ModelMetaclass(type):
    """Stub for V1 ModelMetaclass. Not needed in V2.

    In Pydantic V2, use __pydantic_complete__ or model_post_init instead.
    This is provided only for compatibility during migration.
    """
    pass


# ModelField is replaced by FieldInfo in V2
# Provide alias for compatibility
ModelField = FieldInfo


# ClassAttribute stub - not needed in V2
class ClassAttribute:
    """Stub for V1 ClassAttribute. Use standard class attributes in V2."""
    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.value = value


# GenericModel is no longer needed - just use Generic with BaseModel
from typing import Generic
GenericModel = BaseModel  # Just use BaseModel with Generic[T]


# SHAPE_LIST constant - used for checking field shapes in V1
# In V2, check annotation directly instead
SHAPE_LIST = 'list'


# ---- Error handling compatibility ----

# V2 uses different error handling
class ErrorWrapper:
    """Stub for V1 ErrorWrapper. Use ValidationError directly in V2."""
    def __init__(self, exc: Exception, loc: tuple) -> None:
        self.exc = exc
        self.loc = loc


def display_errors(errors: list[dict]) -> str:
    """Format validation errors for display."""
    lines = []
    for error in errors:
        loc = '.'.join(str(l) for l in error.get('loc', ()))
        msg = error.get('msg', '')
        lines.append(f"  {loc}: {msg}")
    return '\n'.join(lines)


# ---- Settings compatibility ----

class SettingsError(ValueError):
    """Error raised when settings validation fails."""
    pass


# Settings source types (simplified for V2)
if TYPE_CHECKING:
    from pydantic_settings import (
        EnvSettingsSource,
        PydanticBaseSettingsSource,
    )
    SettingsSourceCallable = PydanticBaseSettingsSource
else:
    EnvSettingsSource = Any
    SettingsSourceCallable = Any


# ---- Extra enum for forbid/allow/ignore ----
class Extra:
    """V1-style Extra enum. Use ConfigDict(extra='...') in V2 instead."""
    allow = 'allow'
    forbid = 'forbid'
    ignore = 'ignore'


# ---- Stub modules for compatibility ----

class _ErrorsModule:
    """Stub for pydantic.v1.errors module."""

    class PydanticValueError(ValueError):
        """Base class for pydantic value errors."""
        code = 'value_error'
        msg_template = 'value error'

        def __init__(self, **ctx: Any) -> None:
            self.ctx = ctx
            super().__init__(self.msg_template.format(**ctx))

    class PydanticTypeError(TypeError):
        """Base class for pydantic type errors."""
        code = 'type_error'
        msg_template = 'type error'

        def __init__(self, **ctx: Any) -> None:
            self.ctx = ctx
            super().__init__(self.msg_template.format(**ctx))


errors = _ErrorsModule()


class _TypesModule:
    """Stub for pydantic.v1.types module."""

    class ConstrainedInt(int):
        """V1-style constrained int. Use Annotated[int, Field(...)] in V2."""
        strict: bool = False
        gt: int | None = None
        ge: int | None = None
        lt: int | None = None
        le: int | None = None
        multiple_of: int | None = None

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source_type: Any,
            handler,
        ):
            from pydantic_core import core_schema
            return core_schema.no_info_before_validator_function(
                cls._validate,
                core_schema.int_schema(),
            )

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema_, handler):
            json_schema = handler(core_schema_)
            if cls.gt is not None:
                json_schema['exclusiveMinimum'] = cls.gt
            if cls.ge is not None:
                json_schema['minimum'] = cls.ge
            if cls.lt is not None:
                json_schema['exclusiveMaximum'] = cls.lt
            if cls.le is not None:
                json_schema['maximum'] = cls.le
            if cls.multiple_of is not None:
                json_schema['multipleOf'] = cls.multiple_of
            return json_schema

        @classmethod
        def _validate(cls, v: Any) -> int:
            if not isinstance(v, (int, float)):
                raise TypeError('integer required')
            v = int(v)
            if cls.gt is not None and v <= cls.gt:
                raise ValueError(f'must be greater than {cls.gt}')
            if cls.ge is not None and v < cls.ge:
                raise ValueError(f'must be greater than or equal to {cls.ge}')
            if cls.lt is not None and v >= cls.lt:
                raise ValueError(f'must be less than {cls.lt}')
            if cls.le is not None and v > cls.le:
                raise ValueError(f'must be less than or equal to {cls.le}')
            if cls.multiple_of is not None and v % cls.multiple_of != 0:
                raise ValueError(f'must be a multiple of {cls.multiple_of}')
            return v

    class ConstrainedFloat(float):
        """V1-style constrained float. Use Annotated[float, Field(...)] in V2."""
        strict: bool = False
        gt: float | None = None
        ge: float | None = None
        lt: float | None = None
        le: float | None = None
        multiple_of: float | None = None

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source_type: Any,
            handler,
        ):
            from pydantic_core import core_schema
            return core_schema.no_info_before_validator_function(
                cls._validate,
                core_schema.float_schema(),
            )

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema_, handler):
            json_schema = handler(core_schema_)
            if cls.gt is not None:
                json_schema['exclusiveMinimum'] = cls.gt
            if cls.ge is not None:
                json_schema['minimum'] = cls.ge
            if cls.lt is not None:
                json_schema['exclusiveMaximum'] = cls.lt
            if cls.le is not None:
                json_schema['maximum'] = cls.le
            if cls.multiple_of is not None:
                json_schema['multipleOf'] = cls.multiple_of
            return json_schema

        @classmethod
        def _validate(cls, v: Any) -> float:
            if not isinstance(v, (int, float)):
                raise TypeError('float required')
            v = float(v)
            if cls.gt is not None and v <= cls.gt:
                raise ValueError(f'must be greater than {cls.gt}')
            if cls.ge is not None and v < cls.ge:
                raise ValueError(f'must be greater than or equal to {cls.ge}')
            if cls.lt is not None and v >= cls.lt:
                raise ValueError(f'must be less than {cls.lt}')
            if cls.le is not None and v > cls.le:
                raise ValueError(f'must be less than or equal to {cls.le}')
            return v


types = _TypesModule()


class _UtilsModule:
    """Stub for pydantic.v1.utils module."""
    ROOT_KEY = '__root__'

    @staticmethod
    def sequence_like(v: Any) -> bool:
        return isinstance(v, (list, tuple, set, frozenset))


utils = _UtilsModule()


class _MainModule:
    """Stub for pydantic.v1.main module."""
    ModelMetaclass = ModelMetaclass


main = _MainModule()


# color module compatibility
class _ColorModule:
    """Stub for pydantic.v1.color module."""
    Color = Color


color = _ColorModule()


__all__ = (
    # Core V2 exports
    'BaseModel',
    'BaseSettings',
    'ConfigDict',
    'Field',
    'FieldInfo',
    'FilePath',
    'GetCoreSchemaHandler',
    'GetJsonSchemaHandler',
    'JsonSchemaValue',
    'PrivateAttr',
    'SettingsConfigDict',
    'TypeAdapter',
    'ValidationError',
    'core_schema',
    'field_serializer',
    'field_validator',
    'model_serializer',
    'model_validator',
    # Constrained types
    'conlist',
    'constr',
    # Compatibility aliases (deprecated)
    'ROOT_KEY',
    'SHAPE_LIST',
    'ClassAttribute',
    'Color',
    'EnvSettingsSource',
    'ErrorWrapper',
    'Extra',
    'GenericModel',
    'Generic',
    'ModelField',
    'ModelMetaclass',
    'SettingsError',
    'SettingsSourceCallable',
    'color',
    'display_errors',
    'errors',
    'main',
    'parse_obj_as',
    'root_validator',
    'sequence_like',
    'types',
    'utils',
    'validator',
)
