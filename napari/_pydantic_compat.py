"""
This module provides compatibility between pydantic v1 and v2.

Keep using this compatibility module until we stop using any pydantic v1 API functionality.
This can be removed when everything has been migrated to pydantic v2.
"""

# The Pydantic V2 package can access the Pydantic V1 API by importing through `pydantic.v1`.
# See https://docs.pydantic.dev/latest/migration/#continue-using-pydantic-v1-features
from pydantic.v1 import (
    BaseModel,
    BaseSettings,
    Extra,
    Field,
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
