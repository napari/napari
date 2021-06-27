from pathlib import Path
from typing import Any, Optional, Union

from ...utils.translations import trans
from ._napari_settings import NapariSettings


class _SettingsProxy:
    """Backwards compatibility layer."""

    def __getattribute__(self, name) -> Any:
        return getattr(get_settings(), name)


SETTINGS: Union[NapariSettings, _SettingsProxy] = _SettingsProxy()


def get_settings(path: Optional[Union[Path, str]] = None) -> NapariSettings:
    """
    Get settings for a given path.

    Parameters
    ----------
    path: Path, optional
        The path to read/write the settings from.

    Returns
    -------
    NapariSettings
        The settings manager.

    Notes
    -----
    The path can only be set once per session.
    """
    global SETTINGS

    if isinstance(SETTINGS, _SettingsProxy):
        config_path = Path(path).resolve() if path else None
        SETTINGS = NapariSettings(config_path=config_path)
    elif path is not None:
        import inspect

        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        raise Exception(
            trans._(
                "The path can only be set once per session. Settings called from {calframe[1][3]}",
                deferred=True,
                calframe=calframe,
            )
        )

    return SETTINGS
