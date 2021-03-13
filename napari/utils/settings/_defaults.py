"""Settings management.
"""

from enum import Enum
from typing import List, Tuple

from pydantic import BaseSettings, Field

from ..events.evented_model import EventedModel
from ..notifications import NotificationSeverity
from ..theme import available_themes


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
        field_schema.update(enum=available_themes())

    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise ValueError('must be a string')

        value = v.lower()
        themes = available_themes()
        if value not in available_themes():
            raise ValueError(f'must be one of {", ".join(themes)}')

        return value


class QtBindingChoice(str, Enum):
    """Python Qt binding to use with the application."""

    pyside2 = 'pyside2'
    pyqt5 = 'pyqt5'


class ApplicationSettings(BaseSettings, EventedModel):
    """Main application settings."""

    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option

    schema_version = (0, 1, 0)

    theme: Theme = Field(
        "dark",
        description="Theme selection.",
    )
    ipy_interactive: bool = Field(
        default=True,
        title='IPython interactive',
        description=(
            r'Use interactive %gui qt event loop when creating '
            'napari Viewers in IPython'
        ),
    )
    first_time: bool = True

    # Window state, geometry and position
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
        env_prefix = 'napari_'
        title = "Application settings"
        use_enum_values = True
        validate_all = True

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


class PluginSettings(BaseSettings, EventedModel):
    """Plugin Settings."""

    schema_version = (0, 1, 0)
    plugins_call_order: List[str] = []

    class Config:
        # Pydantic specific configuration
        env_prefix = 'napari_'
        title = "Plugin settings"
        use_enum_values = True

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version', 'plugins_call_order']


CORE_SETTINGS = [ApplicationSettings, PluginSettings]
