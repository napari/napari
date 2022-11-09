"""
Default base variables for defining configuration paths.

This is used by the translation loader as the settings models require using
the translator before the settings manager is created.
"""

from appdirs import user_config_dir

from packaging import version

from napari._version import __version__

_dev_version_config_dir = 'dev' if version.parse(__version__).is_devrelease else None


_FILENAME = "settings.yaml"
_APPNAME = "Napari"
_APPAUTHOR = "Napari"
_DEFAULT_LOCALE = "en"
_DEFAULT_CONFIG_PATH = (
    user_config_dir(_APPNAME, _APPAUTHOR, _dev_version_config_dir)
    + '/'
    + _FILENAME
)
