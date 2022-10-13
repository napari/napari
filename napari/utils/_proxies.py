import os
import re
import sys
import warnings
from typing import Any, Callable, Generic, TypeVar, Union

import wrapt

from ..utils import misc
from ..utils.translations import trans

_T = TypeVar("_T")


class ReadOnlyWrapper(wrapt.ObjectProxy):
    """
    Disable item and attribute setting with the exception of  ``__wrapped__``.
    """

    def __setattr__(self, name, val):
        if name != '__wrapped__':
            raise TypeError(
                trans._(
                    'cannot set attribute {name}',
                    deferred=True,
                    name=name,
                )
            )

        super().__setattr__(name, val)

    def __setitem__(self, name, val):
        raise TypeError(
            trans._('cannot set item {name}', deferred=True, name=name)
        )


_SUNDER = re.compile('^_[^_]')


class PublicOnlyProxy(wrapt.ObjectProxy, Generic[_T]):
    """Proxy to prevent private attribute and item access, recursively."""

    __wrapped__: _T

    @staticmethod
    def _is_private_attr(name: str) -> bool:
        return name.startswith("_") and not (
            name.startswith('__') and name.endswith('__')
        )

    @staticmethod
    def _private_attr_warning(name: str, typ: str):
        warnings.warn(
            trans._(
                "Private attribute access ('{typ}.{name}') in this context (e.g. inside a plugin widget or dock widget) is deprecated and will be unavailable in version 0.5.0",
                deferred=True,
                name=name,
                typ=typ,
            ),
            category=FutureWarning,
            stacklevel=3,
        )

    @staticmethod
    def _is_called_from_napari():
        """
        Check if the getter or setter is called from inner napari.
        """
        if hasattr(sys, "_getframe"):
            frame = sys._getframe(2)
            return frame.f_code.co_filename.startswith(misc.ROOT_DIR)
        return False

    def __getattr__(self, name: str):
        if self._is_private_attr(name):
            if self._is_called_from_napari():
                return super().__getattr__(name)

            typ = type(self.__wrapped__).__name__

            self._private_attr_warning(name, typ)

            # name = f'{type(self.__wrapped__).__name__}.{name}'
            # raise AttributeError(
            #     trans._(
            #         "Private attribute access ('{typ}.{name}') not allowed in this context.",
            #         deferred=True,
            #         name=name,
            #         typ=typ,
            #     )
            # )
        return self.create(super().__getattr__(name))

    def __setattr__(self, name: str, value: Any):
        if (
            os.environ.get("NAPARI_ENSURE_PLUGIN_MAIN_THREAD", False)
            not in ("0", "False")
            and not in_main_thread()
        ):
            raise RuntimeError(
                "Setting attributes on a napari object is only allowed from the main Qt thread."
            )

        if self._is_private_attr(name):
            if self._is_called_from_napari():
                return super().__setattr__(name, value)

            typ = type(self.__wrapped__).__name__
            self._private_attr_warning(name, typ)

            # raise AttributeError(
            #     trans._(
            #         "Private attribute set ('{typ}.{name}') not allowed in this context.",
            #         deferred=True,
            #         name=name,
            #         typ=typ,
            #     )
            # )

        setattr(self.__wrapped__, name, value)

    def __getitem__(self, key):
        return self.create(super().__getitem__(key))

    def __repr__(self):
        return repr(self.__wrapped__)

    def __dir__(self):
        return [x for x in dir(self.__wrapped__) if not _SUNDER.match(x)]

    @classmethod
    def create(cls, obj: Any) -> Union['PublicOnlyProxy', Any]:
        # restrict the scope of this proxy to napari objects
        mod = getattr(type(obj), '__module__', None) or ''
        if not mod.startswith('napari'):
            return obj
        if isinstance(obj, PublicOnlyProxy):
            return obj  # don't double-wrap
        if callable(obj):
            return CallablePublicOnlyProxy(obj)
        return PublicOnlyProxy(obj)


class CallablePublicOnlyProxy(PublicOnlyProxy[Callable]):
    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)


def in_main_thread_py() -> bool:
    """
    Check if caller is in main python thread.

    Returns
    -------
    thread_flag : bool
        True if we are in the main thread, False otherwise.
    """
    import threading

    return threading.current_thread() == threading.main_thread()


def _in_main_thread() -> bool:
    """
    General implementation of checking if we are in a proper thread.
    If Qt is available and Application is created then assign :py:func:`in_qt_main_thread` to `in_main_thread`.
    If Qt liba are not available then assign :py:func:`in_main_thread_py` to in_main_thread.
    IF Qt libs are available but there is no Application ti wil emmit warning and return result of in_main_thread_py.

    Returns
    -------
    thread_flag : bool
        True if we are in the main thread, False otherwise.
    """

    global in_main_thread
    try:
        from napari._qt.utils import in_qt_main_thread

        res = in_qt_main_thread()
        in_main_thread = in_qt_main_thread
        return res
    except ImportError:
        in_main_thread = in_main_thread_py
        return in_main_thread_py()
    except AttributeError:
        warnings.warn(
            "Qt libs are available but no QtApplication instance is created"
        )
        return in_main_thread_py()


in_main_thread = _in_main_thread
