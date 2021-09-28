import warnings
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

    import inspect

    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    for frame in calframe[1:]:
        if (
            frame.function == "__new__"
            and frame.filename
            == '/home/czaki/Projekty/napari/napari/utils/events/evented_model.py'
            and _SETTINGS is not None
        ):
            # contructor of EventedModel
            break
        if frame.function == "<module>" and str(
            frame.frame.f_locals.get("__package__", "")
        ).startswith("napari."):
            warnings.warn(
                f"using settings in global context:\n{frame.filename}:{frame.lineno} {''.join(frame.code_context)}\n"
                + "#" * 10,
                category=RuntimeWarning,
                stacklevel=2,
            )
            break
    if _SETTINGS is None:
        if path is not _NOT_SET:
            path = Path(path).resolve() if path is not None else None
        _SETTINGS = NapariSettings(config_path=path)
    elif path is not _NOT_SET:
        raise RuntimeError(
            f"The path can only be set once per session. Settings called from {calframe[1][3]}"
        )
    return _SETTINGS
