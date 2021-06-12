"""Settings management."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from pydantic import BaseSettings, Field, ValidationError
from pydantic.env_settings import SettingsSourceCallable
from typing_extensions import TypedDict

from ...utils.shortcuts import default_shortcuts
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
                    "A schema version must be a 3 element tuple or string!",
                    deferred=True,
                )
            )

        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError(
                trans._(
                    "A schema version must be a 3 element tuple or string!",
                    deferred=True,
                )
            )

        for part in parts:
            try:
                int(part)
            except Exception:
                raise ValueError(
                    trans._(
                        "A schema version subparts must be positive integers or parseable as integers!",
                        deferred=True,
                    )
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


class QtBindingChoice(str, Enum):
    """Python Qt binding to use with the application."""

    pyside2 = 'pyside2'
    pyqt5 = 'pyqt5'


def yaml_config_settings_source(
    settings: BaseNapariSettings,
) -> Dict[str, Any]:
    """Load and validate settings coming from configuration file."""
    yaml_settings: Dict[str, Any] = {}
    model_class = settings.__class__

    # This is set by the SettingsManager
    loaded_data = settings._LOADED_DATA
    validate = settings._IGNORE_YAML_SOURCE
    if not validate and loaded_data:
        # This variable prevents recursion when using the model for validation
        model_class._IGNORE_YAML_SOURCE = True

        # Get defaults from the schema
        model_schema = model_class.schema()
        section = model_schema["section"]
        default_properties = model_schema.get("properties", {})
        yaml_settings = loaded_data.get(section, {}).copy()

        try:
            model_class(**yaml_settings)
        except ValidationError as e:
            # Handle extra fields
            model_data_replace = {}
            for error in e.errors():
                # Grab the first error entry
                item = error["loc"][0]
                try:
                    model_data_replace[item] = default_properties[item][
                        "default"
                    ]
                except KeyError:
                    yaml_settings.pop(item)

            yaml_settings.update(model_data_replace)

        # This variable restores the normal sources loading behavior
        model_class._IGNORE_YAML_SOURCE = False

    return yaml_settings


class ManagerMixin:
    _IGNORE_YAML_SOURCE: bool = False
    _LOADED_DATA: Dict[str, Any] = {}


class BaseNapariSettings(BaseSettings, EventedModel, ManagerMixin):
    class Config:
        # Pydantic specific configuration
        env_prefix = 'napari_'
        use_enum_values = True
        validate_all = True
        _env_settings: Optional[SettingsSourceCallable] = None

        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ):
            cls._env_settings = env_settings
            return (
                init_settings,
                env_settings,
                yaml_config_settings_source,
                file_secret_settings,
            )


class AppearanceSettings(BaseNapariSettings):
    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option
    schema_version: Union[SchemaVersion, Tuple[int, int, int]] = (0, 1, 1)

    theme: Theme = Field(
        "dark",
        title=trans._("Theme"),
        description=trans._("Select the user interface theme."),
    )

    highlight_thickness: int = Field(
        1,
        title=trans._("Highlight thickness"),
        description=trans._(
            "Select the highlight thickness when hovering over shapes/points."
        ),
        ge=1,
        le=10,
    )

    layer_tooltip_visibility: bool = Field(
        False,
        description=trans._("If layer tooltip will be shown when hower mouse"),
    )

    class Config:
        # Pydantic specific configuration
        schema_extra = {
            "title": trans._("Appearance"),
            "description": trans._("User interface appearance settings."),
            "section": "appearance",
        }

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version']


class ApplicationSettings(BaseNapariSettings):
    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option
    schema_version: Union[SchemaVersion, Tuple[int, int, int]] = (0, 1, 1)
    first_time: bool = Field(
        True,
        title=trans._('First time'),
        description=trans._(
            'Indicate if napari is running for the first time. This setting is managed by the application.'
        ),
    )
    ipy_interactive: bool = Field(
        True,
        title=trans._('IPython interactive'),
        description=trans._(
            r'Toggle the use of interactive `%gui qt` event loop when creating napari Viewers in IPython.'
        ),
    )
    language: Language = Field(
        _DEFAULT_LOCALE,
        title=trans._("Language"),
        description=trans._(
            "Select the display language for the user interface."
        ),
    )
    # Window state, geometry and position
    save_window_geometry: bool = Field(
        True,
        title=trans._("Save window geometry"),
        description=trans._(
            "Toggle saving the main window size and position."
        ),
    )
    save_window_state: bool = Field(
        True,
        title=trans._("Save Window State"),
        description=trans._("Save window state of dock widgets."),
    )
    window_position: Tuple[int, int] = Field(
        None,
        title=trans._("Window position"),
        description=trans._(
            "Last saved x and y coordinates for the main window. This setting is managed by the application."
        ),
    )
    window_size: Tuple[int, int] = Field(
        None,
        title=trans._("Window size"),
        description=trans._(
            "Last saved width and height for the main window. This setting is managed by the application."
        ),
    )
    window_maximized: bool = Field(
        None,
        title=trans._("Window maximized state"),
        description=trans._(
            "Last saved maximized state for the main window. This setting is managed by the application."
        ),
    )
    window_fullscreen: bool = Field(
        None,
        title=trans._("Window fullscreen"),
        description=trans._(
            "Last saved fullscreen state for the main window. This setting is managed by the application."
        ),
    )
    window_state: str = Field(
        None,
        title=trans._("Window state"),
        description=trans._(
            "Last saved state of dockwidgets and toolbars for the main window. This setting is managed by the application."
        ),
    )
    window_statusbar: bool = Field(
        True,
        title=trans._("Show status bar"),
        description=trans._(
            "Toggle diplaying the status bar for the main window."
        ),
    )
    preferences_size: Tuple[int, int] = Field(
        None,
        title=trans._("Preferences size"),
        description=trans._(
            "Last saved width and height for the preferences dialog. This setting is managed by the application."
        ),
    )
    gui_notification_level: NotificationSeverity = Field(
        NotificationSeverity.INFO,
        title=trans._("GUI notification level"),
        description=trans._(
            "Select the notification level for the user interface."
        ),
    )
    console_notification_level: NotificationSeverity = Field(
        NotificationSeverity.NONE,
        title=trans._("Console notification level"),
        description=trans._("Select the notification level for the console."),
    )
    open_history: List[str] = Field(
        [],
        title=trans._("Opened folders history"),
        description=trans._(
            "Last saved list of opened folders. This setting is managed by the application."
        ),
    )
    save_history: List[str] = Field(
        [],
        title=trans._("Saved folders history"),
        description=trans._(
            "Last saved list of saved folders. This setting is managed by the application."
        ),
    )

    class Config:
        # Pydantic specific configuration
        schema_extra = {
            "title": trans._("Application"),
            "description": trans._("Main application settings."),
            "section": "application",
        }

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
            "open_history",
            "save_history",
            "ipy_interactive",
        ]


class PluginHookOption(TypedDict):
    """Custom type specifying plugin and enabled state."""

    plugin: str
    enabled: bool


CallOrderDict = Dict[str, List[PluginHookOption]]


class ShortcutsSettings(BaseNapariSettings):
    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option
    schema_version: Union[SchemaVersion, Tuple[int, int, int]] = (0, 1, 1)
    shortcuts: Dict[str, List[str]] = Field(
        default_shortcuts,
        title=trans._("shortcuts"),
        description=trans._(
            "Sort plugins for each action in the order to be called.",
        ),
    )

    class Config:
        # Pydantic specific configuration
        schema_extra = {
            "title": trans._("Shortcuts"),
            "description": trans._("Shortcut settings."),
            "section": "shortcuts",
        }

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version', 'shortcuts']


class PluginsSettings(BaseNapariSettings):
    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option
    schema_version: Union[SchemaVersion, Tuple[int, int, int]] = (0, 1, 1)
    call_order: CallOrderDict = Field(
        None,
        title=trans._("Plugin sort order"),
        description=trans._(
            "Sort plugins for each action in the order to be called.",
        ),
    )

    disabled_plugins: Set[str] = set()

    class Config:
        # Pydantic specific configuration
        schema_extra = {
            "title": trans._("Plugins"),
            "description": trans._("Plugins settings."),
            "section": "plugins",
        }

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version', 'disabled_plugins']


class ExperimentalSettings(BaseNapariSettings):
    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option
    schema_version: Union[SchemaVersion, Tuple[int, int, int]] = (0, 1, 0)
    octree: Union[bool, str] = Field(
        False,
        title=trans._("Enable Asynchronous Tiling of Images"),
        description=trans._(
            "Renders images asynchronously using tiles. \nYou must restart napari for changes of this setting to apply."
        ),
        type='boolean',  # need to specify to build checkbox in preferences.
    )

    async_: bool = Field(
        False,
        title=trans._("Render Images Asynchronously"),
        description=trans._(
            "Asynchronous loading of image data. \nThis setting partially loads data while viewing. \nYou must restart napari for changes of this setting to apply."
        ),
        env="napari_async",
    )

    class Config:
        # Pydantic specific configuration
        schema_extra = {
            "title": trans._("Experimental"),
            "description": trans._("Experimental settings."),
            "section": "experimental",
        }

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version']


SettingsType = Tuple[
    Type[AppearanceSettings],
    Type[ApplicationSettings],
    Type[PluginsSettings],
    Type[ShortcutsSettings],
    Type[ExperimentalSettings],
]
CORE_SETTINGS: SettingsType = (
    AppearanceSettings,
    ApplicationSettings,
    PluginsSettings,
    ShortcutsSettings,
    ExperimentalSettings,
)
