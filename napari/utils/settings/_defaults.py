"""Settings management.
"""

from enum import Enum
from typing import List, Tuple

from pydantic import BaseSettings, Field

from .._base import _DEFAULT_LOCALE
from ..events.evented_model import EventedModel
from ..notifications import NotificationSeverity
from ..theme import available_themes
from ..translations import _load_language, get_language_packs, trans


class Theme(str):
    """
    Custom theme type to dynamically load all installed themes.
    """

    # https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        # TODO: Provide a way to handle keys so we can display human readable
        # option in the preferences dropdown
        field_schema.update(enum=available_themes())

    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise ValueError(trans._('must be a string'))

        value = v.lower()
        themes = available_themes()
        if value not in available_themes():
            raise ValueError(
                trans._(
                    '"{value}" is not valid. It must be one of {themes}'
                ).format(value=value, themes=", ".join(themes))
            )

        return value


class Language(str):
    """
    Custom theme type to dynamically load all installed language packs.
    """

    # https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        # TODO: Provide a way to handle keys so we can display human readable
        # option in the preferences dropdown
        language_packs = list(get_language_packs(_load_language()).keys())
        field_schema.update(enum=language_packs)

    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise ValueError(trans._('must be a string'))

        language_packs = list(get_language_packs(_load_language()).keys())
        if v not in language_packs:
            raise ValueError(
                trans._(
                    '"{value}" is not valid. It must be one of {language_packs}.'
                ).format(value=v, language_packs=", ".join(language_packs))
            )

        return v


class QtBindingChoice(str, Enum):
    """Python Qt binding to use with the application."""

    pyside2 = 'pyside2'
    pyqt5 = 'pyqt5'


class BaseNapariSettings(BaseSettings, EventedModel):
    class Config:
        # Pydantic specific configuration
        env_prefix = 'napari_'
        use_enum_values = True
        validate_all = True


class AppearanceSettings(BaseNapariSettings):
    """Appearance Settings."""

    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option

    schema_version = (0, 1, 0)

    theme: Theme = Field(
        "dark",
        title=trans._("Theme"),
        description=trans._("Theme selection."),
    )

    class Config:
        # Pydantic specific configuration
        title = trans._("Appearance")

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version']


class ApplicationSettings(BaseNapariSettings):
    """Main application settings."""

    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option

    schema_version = (0, 1, 0)

    first_time: bool = True

    ipy_interactive: bool = Field(
        default=True,
        title=trans._('IPython interactive'),
        description=(
            r'Use interactive %gui qt event loop when creating '
            'napari Viewers in IPython'
        ),
    )

    language: Language = Field(
        _DEFAULT_LOCALE,
        title=trans._("Language"),
        description=trans._("Interface display language."),
    )

    # Window state, geometry and position
    save_window_geometry: bool = Field(
        True,
        title=trans._("Save Window Geometry"),
        description=trans._("Save window size and position."),
    )
    window_position: Tuple[int, int] = None
    window_size: Tuple[int, int] = None
    window_maximized: bool = None
    window_fullscreen: bool = None
    window_state: str = None
    window_statusbar: bool = True
    preferences_size: Tuple[int, int] = None
    # TODO: Might be breaking preferences?
    gui_notification_level: NotificationSeverity = NotificationSeverity.INFO
    console_notification_level: NotificationSeverity = (
        NotificationSeverity.NONE
    )

    class Config:
        # Pydantic specific configuration
        title = trans._("Application")

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = [
            "schema_version",
            "preferences_size",
            "first_time",
            "window_position",
            "window_size",
            "window_maximized",
            "window_fullscreen",
            "window_state",
            "window_statusbar",
            "gui_notification_level",
            "console_notification_level",
        ]


class PluginsSettings(BaseNapariSettings):
    """Plugins Settings."""

    schema_version = (0, 1, 0)
    plugins_call_order: List[str] = Field(
        [],
        title=trans._("Plugin call order"),
        description=trans._("Sort plugins call order"),
    )

    class Config:
        # Pydantic specific configuration
        title = trans._("Plugins")

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version', 'plugins_call_order']


CORE_SETTINGS = [AppearanceSettings, ApplicationSettings, PluginsSettings]
