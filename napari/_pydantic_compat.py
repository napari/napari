try:
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
except ImportError:
    # pydantic v1
    from pydantic import (
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
    from pydantic.env_settings import (
        EnvSettingsSource,
        SettingsError,
        SettingsSourceCallable,
    )
    from pydantic.error_wrappers import ErrorWrapper, display_errors
    from pydantic.fields import SHAPE_LIST, ModelField
    from pydantic.generics import GenericModel
    from pydantic.main import ClassAttribute, ModelMetaclass
    from pydantic.utils import ROOT_KEY, sequence_like

Color = color.Color

__all__ = (
    'BaseModel',
    'BaseSettings',
    'ClassAttribute',
    'Color',
    'EnvSettingsSource',
    'ErrorWrapper',
    'Extra',
    'Field',
    'ModelField',
    'GenericModel',
    'ModelMetaclass',
    'PositiveInt',
    'PrivateAttr',
    'ROOT_KEY',
    'SettingsError',
    'SettingsSourceCallable',
    'SHAPE_LIST',
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
