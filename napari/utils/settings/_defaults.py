"""Settings management.
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseSettings, Field
from typing_extensions import TypedDict

from .._base import _DEFAULT_LOCALE
from ..events.evented_model import EventedModel
from ..notifications import NotificationSeverity
from ..theme import available_themes
from ..translations import _load_language, get_language_packs, trans


class SchemaVersion(str):
    """
    Custom schema version type to handle both tuples and version strings.

    Provides also a `as_tuple` method for convenience when doing version
    comparison.
    """

    def __new__(cls, value):
        if isinstance(value, (tuple, list)):
            value = ".".join(str(item) for item in value)

        return str.__new__(cls, value)

    def __init__(self, value):
        if isinstance(value, (tuple, list)):
            value = ".".join(str(item) for item in value)

        self._value = value

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, (tuple, list)):
            v = ".".join(str(item) for item in v)

        if not isinstance(v, str):
            raise ValueError(
                trans._(
                    "A schema version must be a 3 element tuple or string!"
                ),
                deferred=True,
            )

        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError(
                trans._(
                    "A schema version must be a 3 element tuple or string!"
                ),
                deferred=True,
            )

        for part in parts:
            try:
                int(part)
            except Exception:
                raise ValueError(
                    trans._(
                        "A schema version subparts must be positive integers or parseable as integers!"
                    ),
                    deferred=True,
                )

        return cls(v)

    def __repr__(self):
        return f'SchemaVersion("{self._value}")'

    def __str__(self):
        return f'"{self._value}"'

    def as_tuple(self):
        return tuple(int(p) for p in self._value.split('.'))


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
            raise ValueError(trans._('must be a string', deferred=True))

        value = v.lower()
        themes = available_themes()
        if value not in available_themes():
            raise ValueError(
                trans._(
                    '"{value}" is not valid. It must be one of {themes}',
                    deferred=True,
                    value=value,
                    themes=", ".join(themes),
                )
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
            raise ValueError(trans._('must be a string', deferred=True))

        language_packs = list(get_language_packs(_load_language()).keys())
        if v not in language_packs:
            raise ValueError(
                trans._(
                    '"{value}" is not valid. It must be one of {language_packs}.',
                    deferred=True,
                    value=v,
                    language_packs=", ".join(language_packs),
                )
            )

        return v


class HighlightThickness(int):
    """Highlight thickness to use when hovering over shapes/points."""

    highlight_thickness = 1
    minimum = 1
    maximum = 10


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

    schema_version: SchemaVersion = (0, 1, 0)

    theme: Theme = Field(
        "dark",
        title=trans._("Theme"),
        description=trans._("Theme selection."),
    )

    highlight_thickness: HighlightThickness = Field(
        1,
        description=trans._(
            "Customize the highlight weight indicating selected shapes and points."
        ),
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

    schema_version: SchemaVersion = (0, 1, 0)

    first_time: bool = True

    ipy_interactive: bool = Field(
        default=True,
        title=trans._('IPython interactive'),
        description=trans._(
            r'Use interactive %gui qt event loop when creating napari Viewers in IPython'
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
    save_window_state: bool = Field(
        True,
        title=trans._("Save Window State"),
        description=trans._("Save window state of dock widgets."),
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

    open_history: List = [os.path.dirname(Path.home())]
    save_history: List = [os.path.dirname(Path.home())]

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
            "open_history",
            "save_history",
            "ipy_interactive",
        ]


class PluginHookOption(TypedDict):
    """Custom type specifying plugin and enabled state."""

    plugin: str
    enabled: bool


CallOrderDict = Dict[str, List[PluginHookOption]]


class PluginsSettings(BaseNapariSettings):
    """Plugins Settings."""

    schema_version: SchemaVersion = (0, 1, 1)
    call_order: CallOrderDict = Field(
        None,
        title=trans._("Plugin sort order"),
        description=trans._(
            "Sort plugins for each action in the order to be called.",
        ),
    )

    class Config:
        # Pydantic specific configuration
        title = trans._("Plugins")

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version']


CORE_SETTINGS = [AppearanceSettings, ApplicationSettings, PluginsSettings]
