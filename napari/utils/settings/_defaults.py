"""Settings management.
"""

from enum import Enum
from typing import List, Tuple

from pydantic import BaseSettings, Field, validator

from ...utils.events.evented_model import EventedModel
from ...utils.theme import available_themes


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
    # Python
    qt_binding: str = Field(
        QtBindingChoice.pyside2,
        description="Python Qt binding to use with the application.",
    )
    # qt_binding: QtBindingChoice = QtBindingChoice.pyside2
    # UI Elements
    highlight_thickness: int = 1

    theme: str = Field(
        "dark",
        description="Theme selection.",
    )

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

    @validator('theme')
    def theme_must_be_registered(cls, v):
        themes = available_themes()
        if v.lower() not in available_themes():
            raise ValueError(f'must be one of {", ".join(themes)}')

        return v.lower()

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
            "qt_binding",
            "highlight_thickness",
            "opt_in_telemetry",
            "font_plain_family",
            "font_plain_size",
            "font_rich_family",
            "font_rich_size",
            "window_statusbar",
        ]


class PluginSettings(BaseSettings, EventedModel):
    schema_version = (0, 1, 0)
    plugins_call_order: List[str] = []

    class Config:
        # Pydantic specific configuration
        env_prefix = 'napari_'
        title = "Plugin settings"
        use_enum_values = True

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = []


CORE_SETTINGS = [ApplicationSettings, PluginSettings]
