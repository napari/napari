"""
Default base variables for defining configuration paths.

This is used by the translation loader as the settings models require using
the translator before the settings manager is created.
"""

from pathlib import Path

from ._appdirs import user_config_dir

_FILENAME = "settings.yaml"
_DEFAULT_LOCALE = "en"
_DEFAULT_CONFIG_PATH = Path(user_config_dir(), _FILENAME)
