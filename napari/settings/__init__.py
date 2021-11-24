from pathlib import Path
from typing import Any, Optional, Union, cast

from ..utils.translations import trans
from ._base import _NOT_SET
from ._napari_settings import NapariSettings

__all__ = ['NapariSettings', 'get_settings']


class _SettingsProxy:
    """Backwards compatibility layer."""

    def __getattribute__(self, name) -> Any:
        return getattr(get_settings(), name)


# deprecated
SETTINGS = _SettingsProxy()

# private global object
# will be populated on first call of get_settings
_SETTINGS: Optional[NapariSettings] = None


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
        import inspect

        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        raise Exception(
            trans._(
                "The path can only be set once per session. Settings called from {calframe}",
                deferred=True,
                calframe=calframe[1][3],
            )
        )

    return _SETTINGS
