from pathlib import Path
from typing import Any, NoReturn, overload

from napari.settings._base import (
    _NOT_SET,
    _NotSetType,
)
from napari.settings._napari_settings import (
    _CFG_PATH,
    CURRENT_SCHEMA_VERSION,
    NapariSettings,
)
from napari.settings._plugin_config_generator import (
    PluginPreferences,
    plugin_configuration_generator,
)

__all__ = [
    'CURRENT_SCHEMA_VERSION',
    'NapariSettings',
    'get_plugin_settings',
    'get_settings',
]


class _SettingsProxy:
    """Backwards compatibility layer."""

    def __getattribute__(self, name) -> Any:
        return getattr(get_settings(), name)


# deprecated
SETTINGS = _SettingsProxy()

# private global object
# will be populated on first call of get_settings
_SETTINGS: NapariSettings | None = None


def _raise_path_set_twice() -> NoReturn:
    """Raise when a settings path is provided more than once per session."""
    import inspect

    curframe = inspect.currentframe()
    # frames: [this helper, get_settings/get_plugin_settings, ...real caller]
    calframe = inspect.getouterframes(curframe, 2)
    raise RuntimeError(
        'The path can only be set once per session. '
        f'Settings called from {calframe[2][3]}'
    )


def get_settings(path=_NOT_SET) -> NapariSettings:
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
        _raise_path_set_twice()

    return _SETTINGS


_PLUGIN_PREFERENCES: dict[str, PluginPreferences] = {}


def _clear_plugin_settings_cache(*_args, **_kwargs) -> None:
    """Invalidate the plugin-settings cache.

    `get_plugin_settings` caches _PLUGIN_PREFERENCES i.e. only
    builds it on first call.

    This utility clears the dict so that the next call forces a
    rebuild of the plugin-settings cache.

    Used when plugins are enabled/disabled or in tests.
    """
    _PLUGIN_PREFERENCES.clear()


@overload
def get_plugin_settings(
    plugin: None = None,
    path_dir: Path | str | _NotSetType | None = _NOT_SET,
) -> dict[str, PluginPreferences]: ...


@overload
def get_plugin_settings(
    plugin: str,
    path_dir: Path | str | _NotSetType | None = _NOT_SET,
) -> PluginPreferences: ...


def get_plugin_settings(
    plugin: str | None = None,
    path_dir: Path | str | _NotSetType | None = _NOT_SET,
) -> dict[str, PluginPreferences] | PluginPreferences:
    """Get settings for all plugins, or for a single plugin.
    Plugin settings are declared by plugins in their manifest under
    ``contributions.configurations``. Each plugin's settings are persisted
    to their own file (e.g. ``<config dir>/<plugin-name>.yaml``), stored in
    the same directory as napari's own settings file, and are auto-saved
    whenever a value changes.
    Parameters
    ----------
    plugin : str, optional
        The name of the plugin whose settings should be returned. If
        omitted, a mapping of all plugin settings keyed by plugin name
        is returned.
    path_dir : Path, str, optional
        The directory in which plugin settings files are stored. If not
        provided, defaults to the directory containing napari's own
        settings file (honoring ``NAPARI_CONFIG``). The path can only be
        set once per session.
    Returns
    -------
    PluginPreferences or dict of PluginPreferences
        The settings for the requested plugin, or a mapping of all plugin
        settings keyed by plugin name.
    Examples
    --------
    >>> from napari.settings import get_plugin_settings
    >>> settings = get_plugin_settings('my-plugin')
    >>> settings.reader.lazy
    False
    """
    global _PLUGIN_PREFERENCES

    if not _PLUGIN_PREFERENCES:
        if isinstance(path_dir, _NotSetType):
            # same directory as napari's own settings file (``_CFG_PATH``), so
            # plugin settings honor ``NAPARI_CONFIG`` exactly like napari's do
            path_dir = Path(_CFG_PATH).parent if _CFG_PATH else None
        elif path_dir is not None:
            path_dir = Path(path_dir).resolve()

        for key, model in plugin_configuration_generator().items():
            config_path = (
                None if path_dir is None else path_dir / f'{key}.yaml'
            )
            _PLUGIN_PREFERENCES[key] = model(config_path=config_path)

    elif not isinstance(path_dir, _NotSetType):
        _raise_path_set_twice()

    if plugin is not None:
        try:
            return _PLUGIN_PREFERENCES[plugin]
        except KeyError as err:
            raise KeyError(f"Plugin named '{plugin}' does not exist.") from err

    return _PLUGIN_PREFERENCES
