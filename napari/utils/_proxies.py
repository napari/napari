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
        if os.environ.get("NAPARI_ENSURE_PLUGIN_MAIN_THREAD", False):
            from napari._qt.utils import in_qt_main_thread

            if not in_qt_main_thread():
                raise RuntimeError(
                    "Setting attributes on a napari object is only allowed from the main thread."
                )

        if self._is_private_attr(name):
            if self._is_called_from_napari():
                return super().__setattr__(name, value)

            typ = type(self.__wrapped__).__name__
            self._private_attr_warning(name, typ)

            # raise AttributeError

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
