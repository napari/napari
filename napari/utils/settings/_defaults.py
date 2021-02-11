"""Settings management.
"""

from enum import Enum
from typing import List, Tuple

from pydantic import BaseSettings, Field

from napari.utils.events.evented_model import EventedModel


class QtBindingEnum(str, Enum):
    """Python Qt binding to use with the application."""

    pyside = 'pyside2'
    pyqt = 'pyqt5'


class ThemeEnum(str, Enum):
    """Application color theme."""

    dark = 'dark'
    light = 'light'


class ApplicationSettings(BaseSettings, EventedModel):
    """Main application settings."""

    version = (0, 1, 0)
    # Python
    qt_binding: QtBindingEnum = QtBindingEnum.pyside
    # UI Elements
    highlight_thickness: int = 1
    theme: ThemeEnum = ThemeEnum.dark
    # Startup
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
            "preferences_size",
            "first_time",
            "window_position",
            "window_size",
            "window_maximized",
            "window_fullscreen",
            "window_state",
        ]


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


CORE_SETTINGS = [ApplicationSettings, PluginSettings]
