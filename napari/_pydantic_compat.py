"""
This module was added for compatibility with pydantic v1 and v2.
I think that we should keep them until we stop using pydantic.v1 module, and fully use
pydantic 2 module.
"""

# pydantic v2
from pydantic.v1 import (
    BaseModel,
    BaseSettings,
    Extra,
    Field,
    PositiveInt,
    PrivateAttr,
    ValidationError,
    color,
    conlist,
    constr,
    errors,
    main,
    parse_obj_as,
    root_validator,
    types,
    utils,
    validator,
)
from pydantic.v1.env_settings import (
    EnvSettingsSource,
    SettingsError,
    SettingsSourceCallable,
)
from pydantic.v1.error_wrappers import ErrorWrapper, display_errors
from pydantic.v1.fields import SHAPE_LIST, ModelField
from pydantic.v1.generics import GenericModel
from pydantic.v1.main import ClassAttribute, ModelMetaclass
from pydantic.v1.utils import ROOT_KEY, sequence_like

Color = color.Color

__all__ = (
    'ROOT_KEY',
    'SHAPE_LIST',
    'BaseModel',
    'BaseSettings',
    'ClassAttribute',
    'Color',
    'EnvSettingsSource',
    'ErrorWrapper',
    'Extra',
    'Field',
    'GenericModel',
    'ModelField',
    'ModelMetaclass',
    'PositiveInt',
    'PrivateAttr',
    'SettingsError',
    'SettingsSourceCallable',
    'ValidationError',
    'color',
    'conlist',
    'constr',
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
