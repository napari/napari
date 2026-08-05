from pathlib import Path
from typing import Any

from napari.settings._base import (
    _NOT_SET,
)
from napari.settings._napari_settings import (
    CURRENT_SCHEMA_VERSION,
    NapariSettings,
)
from napari.settings._plugin_config_generator import (
    PluginPreferences,
    plugin_configuration_generator,
)
from napari.utils._platformdirs import user_config_dir

__all__ = ['CURRENT_SCHEMA_VERSION', 'NapariSettings', 'get_settings']


class _SettingsProxy:
    """Backwards compatibility layer."""

    def __getattribute__(self, name) -> Any:
        return getattr(get_settings(), name)


# deprecated
SETTINGS = _SettingsProxy()

# private global object
# will be populated on first call of get_settings
_SETTINGS: NapariSettings | None = None


def get_settings(path=_NOT_SET, plugin=None) -> NapariSettings:
    """
    Get settings for a given path.

    Parameters
    ----------
    path : Path, optional
        The path to read/write the settings from.

    Returns
    -------
    SettingsManager
        The settings manager.

    Notes
    -----
    The path can only be set once per session.
    """
    global _SETTINGS

    if _SETTINGS is None:
        if path is not _NOT_SET:
            path = Path(path).resolve() if path is not None else None
        _SETTINGS = NapariSettings(config_path=path)
    elif path is not _NOT_SET:
        import inspect

        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        raise RuntimeError(
            f'The path can only be set once per session. Settings called from {calframe[1][3]}'
        )

    return _SETTINGS


_PLUGIN_PREFERENCES: dict[str, PluginPreferences] = {}


def get_plugin_settings(
    plugin: str | None = None,
    path_dir=_NOT_SET,
) -> dict[str, PluginPreferences] | PluginPreferences:
    global _PLUGIN_PREFERENCES

    if not _PLUGIN_PREFERENCES:
        if path_dir is _NOT_SET:
            path_dir = Path(user_config_dir())
        elif path_dir is not None:
            path_dir = Path(path_dir).resolve()

        for key, model in plugin_configuration_generator().items():
            config_path = (
                None if path_dir is None else path_dir / f'{key}.yaml'
            )
            _PLUGIN_PREFERENCES[key] = model(config_path=config_path)

    elif path_dir is not _NOT_SET:
        import inspect

        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        raise RuntimeError(
            f'The path can only be set once per session. '
            f'Settings called from {calframe[1][3]}'
        )

    if plugin is not None:
        try:
            return _PLUGIN_PREFERENCES[plugin]
        except KeyError as err:
            raise KeyError(f"Plugin named '{plugin}' does not exist.") from err

    return _PLUGIN_PREFERENCES
