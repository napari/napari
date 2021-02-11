"""Settings management.
theme (superseding #943)
PySide2 vs PyQt5 preference (if both installed)
window position/geometry
opt in for telemetry
font size for console
monitor DPI (#820)
call order of plugins (after #937)
default colormaps or color combinations (as discussed in #619)
magic-naming for layer (off by default in #1008)
highlight thickness for shapes / points layers
dask cache size maximum, and dask fusion settings (#1173 (comment))
version updates (has user been asked)
key/mouse bindings
"""

from enum import Enum
from typing import List, Tuple

from pydantic import BaseSettings, Field

from napari.utils.events.evented_model import EventedModel

# If a plugin registers a theme, how does it work to do model generation?
# FIXME: Generating enums on the fly for Themes?


class QtBindingEnum(str, Enum):
    """Python Qt binding to use with the application."""

    pyside = 'pyside2'
    pyqt = 'pyqt5'


class ThemeEnum(str, Enum):
    """Theme to use with the application."""

    dark = 'dark'
    light = 'light'


class ApplicationSettings(BaseSettings, EventedModel):
    """Main application settings."""

    version = (0, 1, 0)
    # Python
    # qt_binding: QtBindingEnum = QtBindingEnum.pyside
    qt_binding: str = Field(
        QtBindingEnum.pyside,
        description="Python Qt binding to use with the application.",
    )
    # qt_binding: str = QtBindingEnum.pyside
    # UI Elements
    highlight_thickness: int = 1
    # theme: str = ThemeEnum.dark
    # theme: ThemeEnum = ThemeEnum.dark
    theme: str = Field(
        ThemeEnum.dark, description="Theme to use with the application."
    )
    # Startup
    # TODO: Make that a date time; so if telemetry ever changes, we can know wether user have accepted before/after?
    # the change , and/or maybe remind them that we are collecting every year or so?
    opt_in_telemetry: bool = Field(
        False, description="Check to enable telemetry measurements"
    )
    first_time: bool = True
    # Fonts
    font_plain_family: str = None
    font_plain_size: int = None
    font_rich_family: str = None
    font_rich_size: int = 12
    # Window state, geometry and position
    window_position: Tuple[int, int] = None
    window_size: Tuple[int, int] = None
    window_maximized: bool = None
    window_fullscreen: bool = None
    window_state: str = None
    window_statusbar: bool = True
    preferences_size: Tuple[int, int] = None

    class Config:
        # Pydantic specific configuration
        env_prefix = 'napari_settings_application_'
        title = "Application settings"
        use_enum_values = True
        validate_all = True

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = [
            "window_position",
            "window_size",
            "window_maximized",
            "window_fullscreen",
            "window_state",
            "first_time",
            "preferences_size",
            "version",
        ]


class ConsoleSettings(BaseSettings, EventedModel):
    # version = (0, 1, 0)
    some_specific_console_config: str = None

    class Config:
        # Pydantic specific configuration
        env_prefix = 'napari_settings_console_'
        title = "Console settings"
        use_enum_values = True

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = []


class PluginSettings(BaseSettings, EventedModel):
    # version = (0, 1, 0)
    plugins_call_order: List[str] = []

    class Config:
        # Pydantic specific configuration
        env_prefix = 'napari_settings_plugins_'
        title = "Plugin settings"
        use_enum_values = True

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = []


# print(ApplicationSettings().json_schema())
# print(ConsoleSettings().json_schema())
# print(PluginSettings().json_schema())
