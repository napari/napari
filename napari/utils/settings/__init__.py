from contextvars import ContextVar
from pathlib import Path
from typing import Any, Optional, Union, cast

from ...utils.translations import trans
from ._napari_settings import NapariSettings

__all__ = ['NapariSettings', 'get_settings']


class _SettingsProxy:
    """Backwards compatibility layer."""

    def __getattribute__(self, name) -> Any:
        import warnings

        warnings.warn(
            trans._(
                "Accessing SETTINGS is deprecated. Please use get_settings()",
                deferred=True,
            ),
            FutureWarning,
        )
        return getattr(get_settings(), name)


# can we deprecate this?
SETTINGS = _SettingsProxy()


_SETTINGS: ContextVar[Optional[NapariSettings]] = ContextVar(
    '_SETTINGS', default=None
)


def get_settings(path: Optional[Union[Path, str]] = None) -> NapariSettings:

    if _SETTINGS.get() is None:
        _SETTINGS.set(NapariSettings(path))
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
    return cast(NapariSettings, _SETTINGS.get())


no_default = "__no_default__"


def get(key=None, default=no_default):
    """Get elements from global settings

    Use '.' for nested access

    Examples
    --------
    >>> from napari.utils import settings
    >>> settings.get('appearance')
    AppearanceSettings(schema_version=SchemaVersion("0.1.1"), theme='dark',
        highlight_thickness=1, layer_tooltip_visibility=False)

    >>>  settings.get('appearance.theme')
    dark

    >>>  settings.get('appearance.theme.x', default=123)
    123

    See Also
    --------
    napari.utils.settings.get
    """
    result = get_settings()
    if not key:
        return result
    if key.startswith('napari'):
        key = key[7:]

    keys = key.strip().split(".")
    for k in keys:
        try:
            result = getattr(result, k)
        except AttributeError:
            if default is not no_default:
                return default
            else:
                raise
    return result


class set:
    """Temporarily set configuration values within a context manager

    Parameters
    ----------
    arg : mapping or None, optional
        A mapping of configuration key-value pairs to set.

    # TODO:
    **kwargs :
        Additional key-value pairs to set. If ``arg`` is provided, values set
        in ``arg`` will be applied before those in ``kwargs``.
        Double-underscores (``__``) in keyword arguments will be replaced with
        ``.``, allowing nested values to be easily set.

    Examples
    --------
    Set ``'appearance.theme'`` in a context

    >>> from napari.utils import settings
    >>> with settings.set({'appearance': {'theme': 'light'}}):
    ...     pass

    Set ``'appearance.theme'`` globally.

    >>> settings.set({'appearance': {'theme': 'light'}})

    See Also
    --------
    napari.utils.settings.get
    """

    def __init__(self, arg: Optional[dict] = None):
        self.settings = get_settings().copy(deep=True)
        self.settings.update(arg or {})
        self.token = _SETTINGS.set(self.settings)

    def __enter__(self):
        return self.settings

    def __exit__(self, type, value, traceback):
        _SETTINGS.reset(self.token)
